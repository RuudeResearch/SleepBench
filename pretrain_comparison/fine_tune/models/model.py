
import os
import sys
sys.path.append('/oak/stanford/groups/jamesz/magnusrk/pretraining_comparison')
from comparison.utils import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


#model classes
class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Shape: (1, max_seq_len, d_model)
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x shape: (batch_size, seq_len, d_model)
        x = x + self.pe[:, :x.size(1), :]
        return x


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
        pooled_output = transformer_output.mean(dim=1)  # Average pooling over the sequence length
        
        return pooled_output

class SleepEventLSTMClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_classes, pooling_head=4, dropout=0.1, max_seq_length=128):
        super(SleepEventLSTMClassifier, self).__init__()
        
        # Define spatial pooling
        #self.spatial_pooling = AttentionPooling(embed_dim, num_heads=pooling_head, dropout=dropout)

        # Set max sequence length
        if max_seq_length is None:
            max_seq_length = 20000
            
        self.positional_encoding = PositionalEncoding(max_seq_length, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        # Transformer encoder for spatial modeling
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, batch_first=True, norm_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # LSTM for temporal modeling
        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(input_size=embed_dim, hidden_size=embed_dim//2, num_layers=num_layers, batch_first=True, dropout=lstm_dropout, bidirectional=True)
        
        # Fully connected layer for sleep stage classification
        self.fc_sleep_stage = nn.Linear(embed_dim, num_classes)

        self.temporal_pooling = AttentionPooling(embed_dim, num_heads=pooling_head, dropout=dropout)

        self.fc_age = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Softplus()  # Ensures smooth, non-negative outputs
        )

        self.fc_death = nn.Linear(embed_dim, 1)

        self.fc_diagnosis = nn.Linear(embed_dim, 12)

    def forward(self, x, mask):
        B, S, E = x.shape        

        # Apply positional encoding and layer normalization
        x = self.positional_encoding(x)
        x = self.layer_norm(x)

        # Apply transformer encoder for spatial modeling
        mask_temporal = mask[:, :]
        x = self.transformer_encoder(x, src_key_padding_mask=mask_temporal)

        # Apply LSTM for temporal modeling
        x, _ = self.lstm(x)  # Shape: (B, S, E)

        # Apply the final fully connected layer for classification
        sleep_stage = self.fc_sleep_stage(x)  # Shape: (B, S, num_classes)

        
        #x_diagnosis = self.temporal_pooling_diagnosis(x, mask_temporal)
        #x_death = self.temporal_pooling_death(x, mask_temporal)
        #x_age = self.temporal_pooling_age(x, mask_temporal)
        x = self.temporal_pooling(x, mask_temporal)
        hazards_death = self.fc_death(x)
        hazards_diagnosis = self.fc_diagnosis(x)
        age = self.fc_age(x)

        return sleep_stage, mask[:, :], age, hazards_diagnosis, hazards_death  # Return mask along temporal dimension