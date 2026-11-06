"""
End-to-end sleep model: CL-style raw-signal encoder + multitask head.

The encoder mirrors the CL pretraining architecture (per-modality tokenizer
followed by a transformer encoder over the temporal token sequence) but is
trained from random initialization jointly with the downstream head.

Input is a full multi-chunk recording: (B, 16, num_chunks * chunk_size)
where chunk_size = sampling_freq * 60 * sampling_duration (5 min @128Hz = 38400).
"""

import math

import torch
import torch.nn as nn
from einops import rearrange


# --------------------------------------------------------------------------- #
# Building blocks (same as the pre-training encoder and the downstream head)
# --------------------------------------------------------------------------- #

class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding (broadcast across the batch)."""

    def __init__(self, max_seq_len, d_model):
        super().__init__()
        position = torch.arange(max_seq_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(1, max_seq_len, d_model)
        pe[0, :, 0::2] = torch.sin(position * div_term)
        pe[0, :, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe)

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class AttentionPooling(nn.Module):
    """Single transformer-layer pooling, returns the mean over valid positions."""

    def __init__(self, input_dim, num_heads=1, dropout=0.1):
        super().__init__()
        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=input_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x, key_padding_mask=None):
        if key_padding_mask is not None:
            if key_padding_mask.size(1) == 1:
                return x.mean(dim=1)
            if key_padding_mask.dtype != torch.bool:
                key_padding_mask = key_padding_mask.to(dtype=torch.bool)
            out = self.transformer_layer(x, src_key_padding_mask=key_padding_mask)
            valid = (~key_padding_mask).float().unsqueeze(-1)
            return (out * valid).sum(dim=1) / valid.sum(dim=1).clamp(min=1)
        out = self.transformer_layer(x)
        return out.mean(dim=1)


def _conv_block(in_ch, out_ch, input_size):
    """One Conv-BN-ELU-LN block matching the CL Tokenizer."""
    return nn.Sequential(
        nn.Conv1d(in_ch, out_ch, kernel_size=5, stride=2, padding=2),
        nn.BatchNorm1d(out_ch),
        nn.ELU(),
        nn.LayerNorm([out_ch, input_size]),
    )


class _ModalityTokenizer(nn.Module):
    """Per-modality CNN tokenizer (matches CL Tokenizer block-for-block)."""

    def __init__(self, in_channels, input_size=640, output_size=128):
        super().__init__()
        self.tokenizer = nn.Sequential(
            _conv_block(in_channels, 4, input_size // 2),
            _conv_block(4, 8, input_size // 4),
            _conv_block(8, 16, input_size // 8),
            _conv_block(16, 32, input_size // 16),
            _conv_block(32, 64, input_size // 32),
            _conv_block(64, 128, input_size // 64),
            nn.AdaptiveAvgPool1d(1),
            nn.Flatten(),
            nn.Linear(128, output_size),
        )

    def forward(self, x):
        return self.tokenizer(x)


class Tokenizer(nn.Module):
    """Wraps four per-modality tokenizers (BAS, RESP, EKG, EMG)."""

    def __init__(self, input_size=640, output_size=128):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tokenizer_BAS = _ModalityTokenizer(8, input_size, output_size)
        self.tokenizer_RESP = _ModalityTokenizer(5, input_size, output_size)
        self.tokenizer_EKG = _ModalityTokenizer(1, input_size, output_size)
        self.tokenizer_EMG = _ModalityTokenizer(2, input_size, output_size)

    def forward(self, x):
        # x: (b, 16, t*input_size)
        b, _, s = x.shape
        t = s // self.input_size

        x_bas = rearrange(x[:, :8, :], 'b ch (t e) -> (b t) ch e', t=t, e=self.input_size)
        x_resp = rearrange(x[:, 8:13, :], 'b ch (t e) -> (b t) ch e', t=t, e=self.input_size)
        x_ekg = rearrange(x[:, 13:14, :], 'b ch (t e) -> (b t) ch e', t=t, e=self.input_size)
        x_emg = rearrange(x[:, 14:16, :], 'b ch (t e) -> (b t) ch e', t=t, e=self.input_size)

        x_bas = rearrange(self.tokenizer_BAS(x_bas), '(b t) e -> b t e', b=b, t=t)
        x_resp = rearrange(self.tokenizer_RESP(x_resp), '(b t) e -> b t e', b=b, t=t)
        x_ekg = rearrange(self.tokenizer_EKG(x_ekg), '(b t) e -> b t e', b=b, t=t)
        x_emg = rearrange(self.tokenizer_EMG(x_emg), '(b t) e -> b t e', b=b, t=t)
        return x_bas, x_resp, x_ekg, x_emg


# --------------------------------------------------------------------------- #
# Encoder (CL-style)
# --------------------------------------------------------------------------- #

class CLStyleEncoder(nn.Module):
    """
    Raw-signal encoder matching the CL pretraining model:
      - per-modality Tokenizer
      - sinusoidal positional encoding over the 60 tokens of a 5-min chunk
      - shared TransformerEncoder applied per modality
      - concatenate (BAS, RESP, EKG, EMG) along the embedding dim -> 4 * D

    Output for a chunk of shape (b, 16, chunk_samples) is
    (b, tokens_per_chunk, 4 * output_size).
    """

    def __init__(self, input_size=640, output_size=128, num_heads=8, num_layers=6,
                 tokens_per_chunk=60, dropout=0.1):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.tokens_per_chunk = tokens_per_chunk

        self.tokenizer = Tokenizer(input_size=input_size, output_size=output_size)
        self.pos_embedding = PositionalEncoding(max_seq_len=tokens_per_chunk, d_model=output_size)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=output_size,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

    @property
    def embed_dim(self):
        return self.output_size * 4

    def forward(self, x):
        # x: (b, 16, t * input_size)  -- t = tokens_per_chunk
        x_bas, x_resp, x_ekg, x_emg = self.tokenizer(x)

        x_bas = self.transformer_encoder(self.pos_embedding(x_bas))
        x_resp = self.transformer_encoder(self.pos_embedding(x_resp))
        x_ekg = self.transformer_encoder(self.pos_embedding(x_ekg))
        x_emg = self.transformer_encoder(self.pos_embedding(x_emg))

        return torch.cat([x_bas, x_resp, x_ekg, x_emg], dim=2)


# --------------------------------------------------------------------------- #
# Head (mirrors fine_tune SleepEventLSTMClassifier)
# --------------------------------------------------------------------------- #

class MultiTaskHead(nn.Module):
    """Transformer + BiLSTM head with per-token sleep-stage logits and
    pooled outputs for age, AHI, diagnosis hazards, and death hazard."""

    def __init__(self, embed_dim, num_heads=4, num_layers=1, num_classes=5,
                 pooling_head=4, dropout=0.3, max_seq_length=8196, num_diagnosis=12):
        super().__init__()
        self.positional_encoding = PositionalEncoding(max_seq_length, embed_dim)
        self.layer_norm = nn.LayerNorm(embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=num_heads,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        lstm_dropout = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(
            input_size=embed_dim,
            hidden_size=embed_dim // 2,
            num_layers=num_layers,
            batch_first=True,
            dropout=lstm_dropout,
            bidirectional=True,
        )

        self.temporal_pooling = AttentionPooling(embed_dim, num_heads=pooling_head, dropout=dropout)

        self.fc_sleep_stage = nn.Linear(embed_dim, num_classes)
        self.fc_age = nn.Sequential(nn.Linear(embed_dim, 1), nn.Softplus())
        self.fc_death = nn.Linear(embed_dim, 1)
        self.fc_diagnosis = nn.Linear(embed_dim, num_diagnosis)
        self.fc_ahi_diagnosis = nn.Linear(embed_dim, 1)

    def forward(self, x, mask):
        x = self.positional_encoding(x)
        x = self.layer_norm(x)

        x = self.transformer_encoder(x, src_key_padding_mask=mask)
        x, _ = self.lstm(x)

        sleep_stage = self.fc_sleep_stage(x)

        pooled = self.temporal_pooling(x, mask)
        age = self.fc_age(pooled)
        hazards_death = self.fc_death(pooled)
        hazards_diagnosis = self.fc_diagnosis(pooled)
        ahi_diagnosis = self.fc_ahi_diagnosis(pooled)

        return sleep_stage, mask, age, hazards_diagnosis, hazards_death, ahi_diagnosis


# --------------------------------------------------------------------------- #
# Full end-to-end model
# --------------------------------------------------------------------------- #

class EndToEndSleepModel(nn.Module):
    """
    Composes the CL-style encoder with the multitask head and runs the encoder
    chunk-by-chunk (each chunk is `tokens_per_chunk * input_size` raw samples,
    e.g. 60 * 640 = 38400 = 5 minutes at 128 Hz).

    Input
    -----
    x_raw : (B, 16, T) raw signal, T padded to a multiple of chunk_samples
    chunk_mask : (B, num_chunks) bool; True means the chunk is padding

    Output
    ------
    sleep_stage      : (B, num_chunks * tokens_per_chunk, num_classes)
    token_mask       : (B, num_chunks * tokens_per_chunk) bool, True = padding
    age              : (B, 1)
    hazards_diagnosis: (B, num_diagnosis)
    hazards_death    : (B, 1)
    ahi_diagnosis    : (B, 1)
    """

    def __init__(
        self,
        # Encoder dims
        input_size=640,
        encoder_output_size=128,
        encoder_num_heads=8,
        encoder_num_layers=6,
        tokens_per_chunk=60,
        encoder_dropout=0.1,
        # Head dims
        head_num_heads=4,
        head_num_layers=1,
        num_classes=5,
        pooling_head=4,
        head_dropout=0.3,
        max_seq_length=8196,
        num_diagnosis=12,
        # Runtime
        encoder_chunks_per_pass=8,
    ):
        super().__init__()
        self.input_size = input_size
        self.tokens_per_chunk = tokens_per_chunk
        self.chunk_samples = tokens_per_chunk * input_size
        self.encoder_chunks_per_pass = encoder_chunks_per_pass

        self.encoder = CLStyleEncoder(
            input_size=input_size,
            output_size=encoder_output_size,
            num_heads=encoder_num_heads,
            num_layers=encoder_num_layers,
            tokens_per_chunk=tokens_per_chunk,
            dropout=encoder_dropout,
        )

        self.head = MultiTaskHead(
            embed_dim=self.encoder.embed_dim,
            num_heads=head_num_heads,
            num_layers=head_num_layers,
            num_classes=num_classes,
            pooling_head=pooling_head,
            dropout=head_dropout,
            max_seq_length=max_seq_length,
            num_diagnosis=num_diagnosis,
        )

    def _encode(self, x_raw):
        """Encode (B, 16, num_chunks * chunk_samples) -> (B, num_chunks * T, D)."""
        b, c, total = x_raw.shape
        assert total % self.chunk_samples == 0, (
            f"Input length {total} must be a multiple of chunk_samples {self.chunk_samples}"
        )
        num_chunks = total // self.chunk_samples

        x = rearrange(
            x_raw,
            'b c (n s) -> (b n) c s',
            n=num_chunks,
            s=self.chunk_samples,
        )

        # Run the encoder on sub-batches of chunks to bound peak memory.
        outputs = []
        step = max(1, self.encoder_chunks_per_pass) * b
        for start in range(0, x.shape[0], step):
            outputs.append(self.encoder(x[start:start + step]))
        emb_chunks = torch.cat(outputs, dim=0)

        emb = rearrange(
            emb_chunks,
            '(b n) t e -> b (n t) e',
            b=b,
            n=num_chunks,
        )
        return emb, num_chunks

    def forward(self, x_raw, chunk_mask=None):
        emb, num_chunks = self._encode(x_raw)

        if chunk_mask is None:
            token_mask = torch.zeros(
                emb.shape[0], emb.shape[1],
                dtype=torch.bool, device=emb.device,
            )
        else:
            # Expand the per-chunk mask to per-token mask (60 tokens per chunk).
            token_mask = chunk_mask.repeat_interleave(self.tokens_per_chunk, dim=1)
            if token_mask.shape[1] != emb.shape[1]:
                # Defensive trim/pad in case of off-by-one
                token_mask = token_mask[:, : emb.shape[1]]

        return self.head(emb, token_mask)
