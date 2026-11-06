"""Encoders used for pre-training: MAE (time/frequency, masked/all patches, and the DAE
variants, which share this architecture) and CL (pairwise / leave-one-out)."""
import math
import random

import torch
import torch.nn as nn
from einops import rearrange

from .utils import get_mask, replace_masked

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