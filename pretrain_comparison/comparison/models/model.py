import torch
import torch.nn as nn
from einops import rearrange, repeat
import math
import random
import os
import pickle
import torch.nn.functional as F
from loguru import logger
from collections import OrderedDict

import sys
sys.path.append('../')
from utils import get_mask, replace_masked

class AttentionPooling(nn.Module):
    def __init__(self, input_dim, num_heads=1, dropout=0.1):
        super(AttentionPooling, self).__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim, 
            nhead=num_heads,
            dropout=dropout,
            batch_first=True
        )

    def forward(self, x, key_padding_mask=None):
        batch_size, seq_len, input_dim = x.size()
        
        if key_padding_mask is not None:
            if key_padding_mask.size(1) == 1:
                return x.mean(dim=1)
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(dtype=torch.bool)
        
            transformer_output = self.transformer_layer(x, src_key_padding_mask=key_padding_mask)
            
            # Invert mask (1 for valid, 0 for padding) and handle the hidden dimension
            attention_mask = (~key_padding_mask).float().unsqueeze(-1)  # [batch_size, seq_len, 1]
            
            # Calculate masked mean
            pooled_output = (transformer_output * attention_mask).sum(dim=1) / attention_mask.sum(dim=1).clamp(min=1)
        else:
            transformer_output = self.transformer_layer(x)
            pooled_output = transformer_output.mean(dim=1)

        return pooled_output


class MaskTokenTemporal(nn.Module):
    def __init__(self, num_masked_tokens, embedding_dim):
        super().__init__()
        self.num_masked_tokens = num_masked_tokens
        
        # Create a learnable masked token
        self.masked_token = nn.Parameter(torch.randn(embedding_dim))

    def generate_src_mask_rand(self, seq_len, batch_size):
        mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
        indices = []
        if self.num_masked_tokens > 0:
            for i in range(batch_size):
                batch_indices = random.sample(range(seq_len), self.num_masked_tokens)
                mask[i, batch_indices] = True
                indices.append(batch_indices)
        return mask, torch.tensor(indices)

    def forward(self, x):
        # x shape: (batch, time, embedding, channels)
        batch_size, seq_len, embedding_dim, num_channels = x.shape
        
        # Generate random mask
        mask, indices = self.generate_src_mask_rand(seq_len, batch_size)
        mask = mask.to(x.device)
        
        # Expand mask to match x dimensions, but keep it the same for all channels
        expanded_mask = mask.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, num_channels)
        
        # Apply masked token
        masked_x = torch.where(expanded_mask, self.masked_token.unsqueeze(-1), x)
        
        return masked_x, mask, indices

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_len, d_model)  # Changed to [1, max_seq_len, d_model]
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe

class ReversibleInstanceNorm():
    def __init__(self, num_channels, num_groups=1):
        self.norm = nn.GroupNorm(num_groups, num_channels)
        
    def forward(self, x):
        return self.norm(x)


class Bottleneck(nn.Module):
    def __init__(self,in_channel,out_channel,expansion,activation,stride=1,padding = 1):
        super(Bottleneck, self).__init__()
        self.stride=stride
        self.conv1 = nn.Conv1d(in_channel,in_channel*expansion,kernel_size = 1)
        self.conv2 = nn.Conv1d(in_channel*expansion,in_channel*expansion,kernel_size = 3, groups = in_channel*expansion,
                               padding=padding,stride = stride)
        self.conv3 = nn.Conv1d(in_channel*expansion,out_channel,kernel_size = 1, stride =1)
        self.b0 = nn.BatchNorm1d(in_channel*expansion)
        self.b1 =  nn.BatchNorm1d(in_channel*expansion)
        self.d = nn.Dropout()
        self.act = activation()
    def forward(self,x):
        if self.stride == 1:
            y = self.act(self.b0(self.conv1(x)))
            y = self.act(self.b1(self.conv2(y)))
            y = self.conv3(y)
            y = self.d(y)
            y = x+y
            return y
        else:
            y = self.act(self.b0(self.conv1(x)))
            y = self.act(self.b1(self.conv2(y)))
            y = self.conv3(y)
            return y


class MBConv(nn.Module):
    def __init__(self,in_channel,out_channels,expansion,layers,activation=nn.ReLU6,stride = 2):
        super(MBConv, self).__init__()
        self.stack = OrderedDict()
        for i in range(0,layers-1):
            self.stack['s'+str(i)] = Bottleneck(in_channel,in_channel,expansion,activation)
            #self.stack['a'+str(i)] = activation()
        self.stack['s'+str(layers+1)] = Bottleneck(in_channel,out_channels,expansion,activation,stride=stride)
        # self.stack['a'+str(layers+1)] = activation()
        self.stack = nn.Sequential(self.stack)
        
        self.bn = nn.BatchNorm1d(out_channels)
    def forward(self,x):
        x = self.stack(x)
        return self.bn(x)


class EffNet(nn.Module):
    
    def __init__(
            self, 
            in_channel, 
            num_additional_features = 0, 
            depth = [1,2,2,3,3,3,3], 
            channels = [32,16,24,40,80,112,192,320,1280],
            dilation = 1,
            stride = 2,
            expansion = 6):
        super(EffNet, self).__init__()
        logger.info(f"depth: {depth}")
        self.stage1 = nn.Conv1d(in_channel, channels[0], kernel_size=3, stride=stride, padding=1,dilation = dilation) #1 conv
        self.b0 = nn.BatchNorm1d(channels[0])
        self.stage2 = MBConv(channels[0], channels[1], expansion, depth[0], stride=2)# 16 #input, output, depth # 3 conv
        self.stage3 = MBConv(channels[1], channels[2], expansion, depth[1], stride=2)# 24 # 4 conv # d 2
        self.Pool = nn.MaxPool1d(3, stride=1, padding=1) # 
        self.stage4 = MBConv(channels[2], channels[3], expansion, depth[2], stride=2)# 40 # 4 conv # d 2
        self.stage5 = MBConv(channels[3], channels[4], expansion, depth[3], stride=2)# 80 # 5 conv # d
        self.stage6 = MBConv(channels[4], channels[5], expansion, depth[4], stride=2)# 112 # 5 conv
        self.stage7 = MBConv(channels[5], channels[6], expansion, depth[5], stride=2)# 192 # 5 conv
        self.stage8 = MBConv(channels[6], channels[7], expansion, depth[6], stride=2)# 320 # 5 conv
        
        self.stage9 = nn.Conv1d(channels[7], channels[8], kernel_size=1)
        self.AAP = nn.AdaptiveAvgPool1d(1)
        self.act = nn.ReLU()
        self.drop = nn.Dropout()
        self.num_additional_features = num_additional_features
        self.fc = nn.Linear(channels[8] + num_additional_features, 128)# changed from 1 to 128
        
        
    def forward(self, x):
        if self.num_additional_features >0:
            x,additional = x
        # N x 8 x 640
        x = self.b0(self.stage1(x))
        # N x 32 x 320
        x = self.stage2(x)
        # N x 16 x 160
        x = self.stage3(x)
        # N x 24 x 80
        x = self.Pool(x)
        # N x 24 x 40
        x = self.stage4(x)
        # N x 40 x 20
        x = self.stage5(x)
        # N x 80 x 10
        x = self.stage6(x)
        # N x 112 x 10
        x = self.Pool(x)
        # N x 192 x 5
        x = self.stage7(x)
        # N x 320 x 3
        x = self.stage8(x)
        x = self.stage9(x)
        # N x 1280 x 3
        x = self.act(self.AAP(x)[:,:,0])
        # N x 1280
        x = self.drop(x)
        if self.num_additional_features >0:
            x = torch.cat((x,additional),1)
        x = self.fc(x)
        # N x 128

        return x
    

class TokenizerEffinet(nn.Module):
    def __init__(self, input_size=640, output_size=128):
        super(TokenizerEffinet, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.tokenizer_BAS = EffNet(in_channel=8, num_additional_features=0, dilation=1, stride=2, expansion=6)
        self.tokenizer_RESP = EffNet(in_channel=5, num_additional_features=0, dilation=1, stride=2, expansion=6)
        self.tokenizer_EKG = EffNet(in_channel=1, num_additional_features=0, dilation=1, stride=2, expansion=6)
        self.tokenizer_EMG = EffNet(in_channel=2, num_additional_features=0, dilation=1, stride=2, expansion=6)

    def forward(self, x):
        # input: (batch, time x embedding, channels)
        #print(f'x: {x.shape}')
        b, ch, s = x.shape
        t = s // self.input_size
        
        # Now split along the channel dimension (dim=1)
        x_bas = x[:, :8, :]     # First 8 channels for BAS
        x_resp = x[:, 8:13, :]  # Next 5 channels for RESP
        x_ekg = x[:, 13:14, :]  # Next 1 channel for EKG
        x_emg = x[:, 14:16, :]  # Final 2 channels for EMG
        
        #print(f'x_bas: {x_bas.shape}, x_resp: {x_resp.shape}, x_ekg: {x_ekg.shape}, x_emg: {x_emg.shape}')

        # Process each group with its respective tokenizer
        # The shapes should now match the input channels of each tokenizer
        x_bas = rearrange(x_bas, 'b ch (t e) -> (b t) ch e', t=t, e=self.input_size)
        x_bas = self.tokenizer_BAS(x_bas)
        x_bas = rearrange(x_bas, '(b t) e -> b t e', b=b, t=t, e=self.output_size)

        x_resp = rearrange(x_resp, 'b ch (t e) -> (b t) ch e', t=t, e=self.input_size)
        x_resp = self.tokenizer_RESP(x_resp)
        x_resp = rearrange(x_resp, '(b t) e -> b t e', b=b, t=t, e=self.output_size)

        x_ekg = rearrange(x_ekg, 'b ch (t e) -> (b t) ch e', t=t, e=self.input_size)
        x_ekg = self.tokenizer_EKG(x_ekg)
        x_ekg = rearrange(x_ekg, '(b t) e -> b t e', b=b, t=t, e=self.output_size)

        x_emg = rearrange(x_emg, 'b ch (t e) -> (b t) ch e', t=t, e=self.input_size)
        x_emg = self.tokenizer_EMG(x_emg)
        x_emg = rearrange(x_emg, '(b t) e -> b t e', b=b, t=t, e=self.output_size)
        
        return x_bas, x_resp, x_ekg, x_emg



class ModifiedVAEDecoder(nn.Module):
    def __init__(self, 
                 input_size=512,  # Input feature dimension
                 output_channels=16,
                 n_blocks=3):  # Number of upsampling blocks
        super().__init__()
        
        # Initial channel and temporal dimension setup
        initial_channels = 64
        initial_temporal = 80
        
        # Calculate the size needed for initial reshape
        self.fc = nn.Linear(input_size, initial_channels * initial_temporal)
        self.initial_channels = initial_channels
        self.initial_temporal = initial_temporal
        
        # Create decoder blocks
        self.decoder_blocks = nn.ModuleList()
        self.upconv_layers = nn.ModuleList()
        
        # Channel progression
        #channels = [initial_channels, 128, 64, 32, 24, 16]
        channels = [initial_channels, 32, 24, 16]
        kernels = [3,3,3,3]
        paddings = [1,1,1,1]
        out_paddings = [1,1,1,1]
        
        # Create upsampling blocks
        for i in range(n_blocks-1):
            self.upconv_layers.append(
                nn.ConvTranspose1d(
                    channels[i],
                    channels[i+1],
                    kernel_size=kernels[i], 
                    stride=2, 
                    padding=paddings[i],
                    output_padding=out_paddings[i]
                )
            )
            self.decoder_blocks.append(
                self.conv_block(channels[i+1], channels[i+1])
            )
        
        # Simplified final upsampling: just 320->640
        self.final_upsample = nn.ConvTranspose1d(
            channels[-2], channels[-1], 
            kernel_size=kernels[-1],
            stride=2,
            padding=paddings[-1],
            output_padding=out_paddings[-1]
        )

        self.final_decoder = self.conv_block(channels[-1], channels[-1])
    
    def conv_block(self, in_channels, out_channels):
        return nn.Sequential(
            nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )
    
    def forward(self, x, channel_means):
        b, t, e = x.shape
        x = rearrange(x, 'b t e -> (b t) e', b=b, t=t)
        #print(f'x shape post rearrange1: {x.shape}')
        
        # Project and reshape
        x = self.fc(x) 
        #print(f'x shape post fc: {x.shape}')
        x = rearrange(x, '(b t) (ch temp) -> (b t) ch temp', b=b, t=t, ch=self.initial_channels, temp=self.initial_temporal)
        #print(f'x shape post rearrange2: {x.shape}')
        
        
        # Apply upsampling blocks
        for i, (upconv, decoder_block) in enumerate(zip(self.upconv_layers, self.decoder_blocks)):
            #print(f'x shape pre upconv {i}: {x.shape}')
            x = upconv(x)
            #print(f'x shape post upconv {i}: {x.shape}')
            x = decoder_block(x)
            #print(f'x shape post decoder_block {i}: {x.shape}')
        
        # Final upsampling to reach 640
        #print(f'x shape pre final_upsample: {x.shape}')
        x = self.final_upsample(x)
        #print(f'x shape post final_upsample: {x.shape}')
        x = rearrange(x, '(b t) ch e -> b ch t e', b=b, t=t, ch=16)
        x = x + channel_means.unsqueeze(-1)
        x = rearrange(x, 'b ch t e-> b ch (t e) ', b=b, t=t, ch=16)
        #x = x + channel_means.view(1, -1, 1)  # Add bias to each channel
        #print(f'x shape post rearrange3: {x.shape}')

        # parameters: initial temporal = 10; initial channels = 256; n_blocks = 6
        # x shape post rearrange1: torch.Size([7680, 512])
        # x shape post fc: torch.Size([7680, 2560])
        # x shape post rearrange2: torch.Size([7680, 256, 10])
        # x shape pre upconv 0: torch.Size([7680, 256, 10])
        # x shape post upconv 0: torch.Size([7680, 128, 20])
        # x shape post decoder_block 0: torch.Size([7680, 128, 20])
        # x shape pre upconv 1: torch.Size([7680, 128, 20])
        # x shape post upconv 1: torch.Size([7680, 64, 40])
        # x shape post decoder_block 1: torch.Size([7680, 64, 40])
        # x shape pre upconv 2: torch.Size([7680, 64, 40])
        # x shape post upconv 2: torch.Size([7680, 32, 80])
        # x shape post decoder_block 2: torch.Size([7680, 32, 80])
        # x shape pre upconv 3: torch.Size([7680, 32, 80])
        # x shape post upconv 3: torch.Size([7680, 24, 160])
        # x shape post decoder_block 3: torch.Size([7680, 24, 160])
        # x shape pre upconv 4: torch.Size([7680, 24, 160])
        # x shape post upconv 4: torch.Size([7680, 16, 320])
        # x shape post decoder_block 4: torch.Size([7680, 16, 320])
        # x shape pre final_upsample: torch.Size([7680, 16, 320])
        # x shape post final_upsample: torch.Size([7680, 16, 640])
        # x shape post rearrange3: torch.Size([128, 16, 38400])

        # parameters: initial temporal = 80; initial channels = 32; n_blocks = 3
        # x shape post rearrange1: torch.Size([7680, 512])
        # x shape post fc: torch.Size([7680, 2560])
        # x shape post rearrange2: torch.Size([7680, 32, 80])
        # x shape pre upconv 0: torch.Size([7680, 32, 80])
        # x shape post upconv 0: torch.Size([7680, 24, 160])
        # x shape post decoder_block 0: torch.Size([7680, 24, 160])
        # x shape pre upconv 1: torch.Size([7680, 24, 160])
        # x shape post upconv 1: torch.Size([7680, 16, 320])
        # x shape post decoder_block 1: torch.Size([7680, 16, 320])
        # x shape pre final_upsample: torch.Size([7680, 16, 320])
        # x shape post final_upsample: torch.Size([7680, 16, 640])
        # x shape post rearrange3: torch.Size([128, 16, 38400])

        
        return x

class DecoderMAE(nn.Module):
    def __init__(self, input_size=int(640*16), output_size=int(128*4)):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        
        self.decoder = nn.Sequential(
            nn.Linear(self.output_size, int(self.output_size*2)),  # Decompress to original chunk size
            nn.ReLU(),
            nn.Linear(int(self.output_size*2), int(self.output_size*2**2)),  # Decompress to original chunk size
            nn.ReLU(),
            nn.Linear(int(self.output_size*2**2), int(self.output_size*2**3)),  # Decompress to original chunk size
            nn.ReLU(),
            nn.Linear(int(self.output_size*2**3), int(self.input_size)), # Further expansion
        )
    
    def forward(self, x):
        b, t, e = x.shape
        x = rearrange(x, 'b t e -> (b t) e', b=b, t=t)
        x = self.decoder(x)
        x = rearrange(x, '(b t) (ch e)-> b ch (t e) ', b=b, t=t, ch=16)

        return x

class SmallDecoderMAE(nn.Module):
    def __init__(self, input_size=int(640*16), output_size=int(128*4)):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        
        self.decoder = nn.Sequential(
            nn.Linear(self.output_size, int(self.output_size*2)),  # Decompress to original chunk size
            nn.ReLU(),
            nn.Linear(int(self.output_size*2), int(self.output_size*2**2)),  # Decompress to original chunk size
            nn.ReLU(),
            nn.Linear(int(self.output_size*2**2), int(self.input_size)), # Further expansion
        )
    
    def forward(self, x):
        b, t, e = x.shape
        x = rearrange(x, 'b t e -> (b t) e', b=b, t=t)
        x = self.decoder(x)
        x = rearrange(x, '(b t) (ch e)-> b ch (t e) ', b=b, t=t, ch=16)

        return x

class TinyDecoderMAE(nn.Module):
    def __init__(self, input_size=int(640*16), output_size=int(128*4)):
        super().__init__()
        self.output_size = output_size
        self.input_size = input_size
        
        self.decoder = nn.Sequential(
            nn.Linear(self.output_size, int(self.input_size)),  # Decompress to original chunk size
        )
        #set the bias of the decoder to the signal mean which is 0
        #self.decoder[0].bias.data = torch.ones(self.input_size) *  0.000043
        # set the bias to 0 
        # self.decoder[0].bias.data = torch.zeros(self.input_size)
        # set the weights to 0
        # self.decoder[0].weight.data = torch.zeros(self.input_size, self.output_size)
    
    def forward(self, x):
        b, t, e = x.shape
        x = rearrange(x, 'b t e -> (b t) e', b=b, t=t)
        x = self.decoder(x)
        x = rearrange(x, '(b t) (ch e)-> b ch (t e) ', b=b, t=t, ch=16)
        #x = x + channel_means.unsqueeze(-1)
        #x = rearrange(x, 'b ch t e-> b ch (t e) ', b=b, t=t, ch=16)

        return x


class Tokenizer(nn.Module):
    def __init__(self, input_size=640, output_size=128):
        super(Tokenizer, self).__init__()
        self.input_size = input_size
        self.output_size = output_size

        self.tokenizer_BAS = nn.Sequential(
            nn.Conv1d(8, 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(4),
            nn.ELU(),
            nn.LayerNorm([4, self.input_size//2]),  
            
            nn.Conv1d(4, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.LayerNorm([8, self.input_size//4]),
            
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.LayerNorm([16, self.input_size//8]),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.LayerNorm([32, self.input_size//16]),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.LayerNorm([64, self.input_size//32]),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.LayerNorm([128, self.input_size//64]),

            # adaptive avg pool 
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),  # Flatten the output
            #nn.Linear(self.input_size//64 * 128, output_size)  # Compress further
            nn.Linear(128, output_size)
        )
        self.tokenizer_RESP = nn.Sequential(
            nn.Conv1d(5, 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(4),
            nn.ELU(),
            nn.LayerNorm([4, self.input_size//2]),  
            
            nn.Conv1d(4, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.LayerNorm([8, self.input_size//4]),
            
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.LayerNorm([16, self.input_size//8]),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.LayerNorm([32, self.input_size//16]),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.LayerNorm([64, self.input_size//32]),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.LayerNorm([128, self.input_size//64]),

            # adaptive avg pool 
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),  # Flatten the output
            #nn.Linear(self.input_size//64 * 128, output_size)  # Compress further
            nn.Linear(128, output_size)
        )
        self.tokenizer_EKG = nn.Sequential(
            nn.Conv1d(1, 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(4),
            nn.ELU(),
            nn.LayerNorm([4, self.input_size//2]),  
            
            nn.Conv1d(4, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.LayerNorm([8, self.input_size//4]),
            
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.LayerNorm([16, self.input_size//8]),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.LayerNorm([32, self.input_size//16]),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.LayerNorm([64, self.input_size//32]),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.LayerNorm([128, self.input_size//64]),

            # adaptive avg pool 
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),  # Flatten the output
            #nn.Linear(self.input_size//64 * 128, output_size)  # Compress further
            nn.Linear(128, output_size)
        )
        self.tokenizer_EMG = nn.Sequential(
            nn.Conv1d(2, 4, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(4),
            nn.ELU(),
            nn.LayerNorm([4, self.input_size//2]),  
            
            nn.Conv1d(4, 8, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(8),
            nn.ELU(),
            nn.LayerNorm([8, self.input_size//4]),
            
            nn.Conv1d(8, 16, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(16),
            nn.ELU(),
            nn.LayerNorm([16, self.input_size//8]),
            
            nn.Conv1d(16, 32, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(32),
            nn.ELU(),
            nn.LayerNorm([32, self.input_size//16]),

            nn.Conv1d(32, 64, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(64),
            nn.ELU(),
            nn.LayerNorm([64, self.input_size//32]),

            nn.Conv1d(64, 128, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm1d(128),
            nn.ELU(),
            nn.LayerNorm([128, self.input_size//64]),

            # adaptive avg pool 
            nn.AdaptiveAvgPool1d(1),

            nn.Flatten(),  # Flatten the output
            #nn.Linear(self.input_size//64 * 128, output_size)  # Compress further
            nn.Linear(128, output_size)
        )

    def forward(self, x):
        # input: (batch, time x embedding, channels)
        #print(f'x: {x.shape}')
        b, ch, s = x.shape
        t = s // self.input_size
        
        # Now split along the channel dimension (dim=1)
        x_bas = x[:, :8, :]     # First 8 channels for BAS
        x_resp = x[:, 8:13, :]  # Next 5 channels for RESP
        x_ekg = x[:, 13:14, :]  # Next 1 channel for EKG
        x_emg = x[:, 14:16, :]  # Final 2 channels for EMG
        
        #print(f'x_bas: {x_bas.shape}, x_resp: {x_resp.shape}, x_ekg: {x_ekg.shape}, x_emg: {x_emg.shape}')

        # Process each group with its respective tokenizer
        # The shapes should now match the input channels of each tokenizer
        x_bas = rearrange(x_bas, 'b ch (t e) -> (b t) ch e', t=t, e=self.input_size)
        x_bas = self.tokenizer_BAS(x_bas)
        x_bas = rearrange(x_bas, '(b t) e -> b t e', b=b, t=t, e=self.output_size)

        x_resp = rearrange(x_resp, 'b ch (t e) -> (b t) ch e', t=t, e=self.input_size)
        x_resp = self.tokenizer_RESP(x_resp)
        x_resp = rearrange(x_resp, '(b t) e -> b t e', b=b, t=t, e=self.output_size)

        x_ekg = rearrange(x_ekg, 'b ch (t e) -> (b t) ch e', t=t, e=self.input_size)
        x_ekg = self.tokenizer_EKG(x_ekg)
        x_ekg = rearrange(x_ekg, '(b t) e -> b t e', b=b, t=t, e=self.output_size)

        x_emg = rearrange(x_emg, 'b ch (t e) -> (b t) ch e', t=t, e=self.input_size)
        x_emg = self.tokenizer_EMG(x_emg)
        x_emg = rearrange(x_emg, '(b t) e -> b t e', b=b, t=t, e=self.output_size)
        
        return x_bas, x_resp, x_ekg, x_emg

class MAE(nn.Module):
    def __init__(self, mask_ratio = 0.34, input_size=640, output_size=128):
        super(MAE, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        max_seq_len = 60 
        self.num_heads = 8
        self.num_layers = 6
        self.mask_ratio = mask_ratio

        self.mask_token_temporal = MaskTokenTemporal(num_masked_tokens = max_seq_len // 3, embedding_dim = output_size)
        self.pos_embedding = PositionalEncoding(max_seq_len = max_seq_len, d_model = output_size)
        #self.norm = ReversibleInstanceNorm(num_channels = output_size, num_groups = 16)

        #self.tokenizer = TokenizerEffinet(input_size=input_size, output_size=output_size)
        self.tokenizer = Tokenizer(input_size=input_size, output_size=output_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_size, nhead=self.num_heads, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        #self.decoder = ModifiedVAEDecoder()
        #self.decoder = DecoderMAE()
        #self.decoder = SmallDecoderMAE()
        #self.decoder = MiniDecoderMAE()
        self.decoder = TinyDecoderMAE()

        #masked token with shape 640
        self.masked_token = nn.Parameter(torch.randn([1,1,1,input_size]))

        #print the number of parameters for each model
        print(f"Number of parameters for tokenizer: {sum(p.numel() for p in self.tokenizer.parameters())}")
        print(f"Number of parameters for transformer_encoder: {sum(p.numel() for p in self.transformer_encoder.parameters())}")
        print(f"Number of parameters for decoder: {sum(p.numel() for p in self.decoder.parameters())}")
    

    def forward(self, x):
        # get the signal mean for setting the bias of the decoder
        mask = get_mask(x, ratio = self.mask_ratio)
        # based on the mask, get the
        x = replace_masked(x, mask, self.masked_token)
        x_bas, x_resp, x_ekg, x_emg = self.tokenizer(x)

        x_bas = self.pos_embedding(x_bas)
        x_resp = self.pos_embedding(x_resp)
        x_ekg = self.pos_embedding(x_ekg)
        x_emg = self.pos_embedding(x_emg)

        x_bas = self.transformer_encoder(x_bas)
        x_resp = self.transformer_encoder(x_resp)
        x_ekg = self.transformer_encoder(x_ekg)
        x_emg = self.transformer_encoder(x_emg)

        x = torch.cat([x_bas, x_resp, x_ekg, x_emg], dim=2)
        embedding = x 
        out = self.decoder(x)
        return out, embedding, mask


class CAE(nn.Module):
    def __init__(self, mask_ratio = 0.34, input_size=640, output_size=128):
        super(CAE, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        max_seq_len = 60 
        self.num_heads = 8
        self.num_layers = 6
        self.mask_ratio = mask_ratio

        self.mask_token_temporal = MaskTokenTemporal(num_masked_tokens = max_seq_len // 3, embedding_dim = output_size)
        self.pos_embedding = PositionalEncoding(max_seq_len = max_seq_len, d_model = output_size)

        self.predictor_pos_embed = PositionalEncoding(max_seq_len = 60, d_model = int(output_size*4))
        predctor_layer = nn.TransformerEncoderLayer(d_model=int(output_size*4), nhead=4, dropout=0.0, batch_first=True, norm_first=True)        
        self.predictor = nn.TransformerEncoder(predctor_layer, num_layers=2)
        #self.norm = ReversibleInstanceNorm(num_channels = output_size, num_groups = 16)

        #self.tokenizer = TokenizerEffinet(input_size=input_size, output_size=output_size)
        self.tokenizer = Tokenizer(input_size=input_size, output_size=output_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_size, nhead=self.num_heads, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        #self.decoder = ModifiedVAEDecoder()
        #self.decoder = DecoderMAE()
        #self.decoder = SmallDecoderMAE()
        #self.decoder = MiniDecoderMAE()
        self.decoder = TinyDecoderMAE()

        #masked token with shape 640
        self.masked_token = nn.Parameter(torch.randn([1,1,1,input_size]))

        #print the number of parameters for each model
        print(f"Number of parameters for tokenizer: {sum(p.numel() for p in self.tokenizer.parameters())}")
        print(f"Number of parameters for transformer_encoder: {sum(p.numel() for p in self.transformer_encoder.parameters())}")
        print(f"Number of parameters for decoder: {sum(p.numel() for p in self.decoder.parameters())}")
    

    def forward(self, x):
        # get the signal mean for setting the bias of the decoder
        with torch.no_grad():
            x_bas_context, x_resp_context, x_ekg_context, x_emg_context = self.tokenizer(x)

            x_bas_context = self.pos_embedding(x_bas_context)
            x_resp_context = self.pos_embedding(x_resp_context)
            x_ekg_context = self.pos_embedding(x_ekg_context)
            x_emg_context = self.pos_embedding(x_emg_context)

            x_bas_context = self.transformer_encoder(x_bas_context)
            x_resp_context = self.transformer_encoder(x_resp_context)
            x_ekg_context = self.transformer_encoder(x_ekg_context)
            x_emg_context = self.transformer_encoder(x_emg_context)

            embedding_context = torch.cat([x_bas_context, x_resp_context, x_ekg_context, x_emg_context], dim=2)
            


        mask = get_mask(x, ratio = self.mask_ratio)
        # based on the mask, get the
        x = replace_masked(x, mask, self.masked_token)
        x_bas, x_resp, x_ekg, x_emg = self.tokenizer(x)

        x_bas = self.pos_embedding(x_bas)
        x_resp = self.pos_embedding(x_resp)
        x_ekg = self.pos_embedding(x_ekg)
        x_emg = self.pos_embedding(x_emg)

        x_bas = self.transformer_encoder(x_bas)
        x_resp = self.transformer_encoder(x_resp)
        x_ekg = self.transformer_encoder(x_ekg)
        x_emg = self.transformer_encoder(x_emg)

        x = torch.cat([x_bas, x_resp, x_ekg, x_emg], dim=2)
        embedding = self.predictor_pos_embed(x)
        embedding = self.predictor(embedding)
        out = self.decoder(embedding)
        return out, embedding, mask, embedding_context


class LearnablePositionalEncoding(nn.Module):
    """
    Learnable positional encoding where each position gets its own embedding vector
    that can be optimized during training
    """
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        # Create a learnable parameter matrix of shape (max_seq_len, d_model)
        # Initialize with small random values to prevent dominating the signal
        self.pos_embedding = nn.Parameter(
            torch.randn(max_seq_len, d_model) * 0.02  # Small initialization scale
        )
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch_size, seq_len, d_model)
        Returns:
            x with positional embeddings added
        """
        # Get the sequence length of the input
        seq_len = x.size(1)
        
        # Add positional embeddings to each position in the sequence
        # The pos_embedding will be automatically broadcast across the batch dimension
        return x + self.pos_embedding[:seq_len, :]

class jepaEncoder(nn.Module):
    def __init__(self, mask_ratio = 0.34, input_size=640, output_size=128):
        super(jepaEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        max_seq_len = 60 
        self.num_heads = 8
        self.num_layers = 6
        self.mask_ratio = mask_ratio

        self.mask_token_temporal = MaskTokenTemporal(num_masked_tokens = max_seq_len // 3, embedding_dim = output_size)
        self.pos_embedding = LearnablePositionalEncoding(max_seq_len = max_seq_len, d_model = output_size)
        #self.norm = ReversibleInstanceNorm(num_channels = output_size, num_groups = 16)

        #self.tokenizer = TokenizerEffinet(input_size=input_size, output_size=output_size)
        self.tokenizer = Tokenizer(input_size=input_size, output_size=output_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_size, nhead=self.num_heads, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        #self.decoder = ModifiedVAEDecoder()
        #self.decoder = DecoderMAE()
        #self.decoder = SmallDecoderMAE()
        #self.decoder = MiniDecoderMAE()
        #self.decoder = TinyDecoderMAE()
        
        #masked token with shape 640
        self.masked_token = nn.Parameter(torch.randn([1,1,1,input_size]))

        #print the number of parameters for each model
        print(f"Number of parameters for tokenizer: {sum(p.numel() for p in self.tokenizer.parameters())}")
        print(f"Number of parameters for transformer_encoder: {sum(p.numel() for p in self.transformer_encoder.parameters())}")
        #print(f"Number of parameters for decoder: {sum(p.numel() for p in self.decoder.parameters())}")

        self._init_weights()

    def _init_weights(self):
        """Initialize weights similar to ViT/BERT style"""
        def _init_layer(module):
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                # Initialize linear layers with truncated normal
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.constant_(module.bias, 0)
                torch.nn.init.constant_(module.weight, 1.0)
                
        self.apply(_init_layer)
    

    def forward(self, x):
        # get the signal mean for setting the bias of the decoder
        #mask = get_mask(x, ratio = masking_ratio)
        # based on the mask, get the
        #x = replace_masked(x, mask, self.masked_token)
        x_bas, x_resp, x_ekg, x_emg = self.tokenizer(x)

        x_bas = self.pos_embedding(x_bas)
        x_resp = self.pos_embedding(x_resp)
        x_ekg = self.pos_embedding(x_ekg)
        x_emg = self.pos_embedding(x_emg)

        x_bas = self.transformer_encoder(x_bas)
        x_resp = self.transformer_encoder(x_resp)
        x_ekg = self.transformer_encoder(x_ekg)
        x_emg = self.transformer_encoder(x_emg)

        x = torch.cat([x_bas, x_resp, x_ekg, x_emg], dim=2)
        return None, x, None #returns the embedding

class dinoEncoder(nn.Module):
    def __init__(self, mask_ratio = 0.34, input_size=640, output_size=128):
        super(dinoEncoder, self).__init__()
        self.input_size = input_size
        self.output_size = output_size
        max_seq_len = 60 
        self.num_heads = 8
        self.num_layers = 6
        self.mask_ratio = mask_ratio

        self.mask_token_temporal = MaskTokenTemporal(num_masked_tokens = max_seq_len // 3, embedding_dim = output_size)
        self.pos_embedding = LearnablePositionalEncoding(max_seq_len = max_seq_len, d_model = output_size)
        #self.norm = ReversibleInstanceNorm(num_channels = output_size, num_groups = 16)

        #self.tokenizer = TokenizerEffinet(input_size=input_size, output_size=output_size)
        self.tokenizer = Tokenizer(input_size=input_size, output_size=output_size)
        encoder_layer = nn.TransformerEncoderLayer(d_model=output_size, nhead=self.num_heads, dropout=0.1, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=self.num_layers)
        #self.decoder = ModifiedVAEDecoder()
        #self.decoder = DecoderMAE()
        #self.decoder = SmallDecoderMAE()
        #self.decoder = MiniDecoderMAE()
        #self.decoder = TinyDecoderMAE()
        
        #masked token with shape 640
        self.masked_token = nn.Parameter(torch.randn([1,1,1,input_size]))

        self.CLS_pooling = AttentionPooling(int(output_size * 4), num_heads=4, dropout=0.0)

        #print the number of parameters for each model
        print(f"Number of parameters for tokenizer: {sum(p.numel() for p in self.tokenizer.parameters())}")
        print(f"Number of parameters for transformer_encoder: {sum(p.numel() for p in self.transformer_encoder.parameters())}")
        #print(f"Number of parameters for decoder: {sum(p.numel() for p in self.decoder.parameters())}")

        self._init_weights()

    def _init_weights(self):
        """Initialize weights similar to ViT/BERT style"""
        def _init_layer(module):
            if isinstance(module, (nn.Linear, nn.Conv1d)):
                # Initialize linear layers with truncated normal
                torch.nn.init.trunc_normal_(module.weight, std=0.02)
                if module.bias is not None:
                    torch.nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.LayerNorm):
                torch.nn.init.constant_(module.bias, 0)
                torch.nn.init.constant_(module.weight, 1.0)
                
        self.apply(_init_layer)
    

    def forward(self, x):
        # get the signal mean for setting the bias of the decoder
        #mask = get_mask(x, ratio = masking_ratio)
        # based on the mask, get the
        #x = replace_masked(x, mask, self.masked_token)
        x_bas, x_resp, x_ekg, x_emg = self.tokenizer(x)

        x_bas = self.pos_embedding(x_bas)
        x_resp = self.pos_embedding(x_resp)
        x_ekg = self.pos_embedding(x_ekg)
        x_emg = self.pos_embedding(x_emg)

        x_bas = self.transformer_encoder(x_bas)
        x_resp = self.transformer_encoder(x_resp)
        x_ekg = self.transformer_encoder(x_ekg)
        x_emg = self.transformer_encoder(x_emg)

        x = torch.cat([x_bas, x_resp, x_ekg, x_emg], dim=2)
        cls_token = self.CLS_pooling(x)
        return cls_token, x, None #returns the embedding



class CL(nn.Module):
    def __init__(self, input_size=640, output_size=128, num_heads=8, num_layers=6, max_seq_len=60, dropout=0.1):
        super(CL, self).__init__()

        self.pos_embedding = PositionalEncoding(max_seq_len=max_seq_len, d_model=output_size)

        self.tokenizer = Tokenizer(input_size=input_size, output_size=output_size)

        encoder_layer = nn.TransformerEncoderLayer(d_model=output_size, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.temporal_pooling = AttentionPooling(output_size, num_heads=num_heads, dropout=dropout)


    def forward(self, x):

        x_bas, x_resp, x_ekg, x_emg = self.tokenizer(x)

        x_bas = self.pos_embedding(x_bas)
        x_resp = self.pos_embedding(x_resp)
        x_ekg = self.pos_embedding(x_ekg)
        x_emg = self.pos_embedding(x_emg)

        x_bas = self.transformer_encoder(x_bas)
        x_resp = self.transformer_encoder(x_resp)
        x_ekg = self.transformer_encoder(x_ekg)
        x_emg = self.transformer_encoder(x_emg)

        x_bas_pooled = self.temporal_pooling(x_bas)
        x_resp_pooled = self.temporal_pooling(x_resp)
        x_ekg_pooled = self.temporal_pooling(x_ekg)
        x_emg_pooled = self.temporal_pooling(x_emg)

        embedding = torch.cat([x_bas, x_resp, x_ekg, x_emg], dim=2)
        out = (x_bas_pooled, x_resp_pooled, x_ekg_pooled, x_emg_pooled)

        return out, embedding, None