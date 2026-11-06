"""Losses, masking, noise generation and file helpers for pre-training."""
import json
import pickle
import random
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import yaml
from einops import rearrange
from scipy.signal import filtfilt, butter


    
class NoiseGenerator:
    NOISE_TYPES = ['white', 'respiratory', 'emg', 'burst']# add only white?

    def __init__(self, num_types=3, noise_limits=(0.01, 0.3), signal_frequency=128, noise_channel_prob=0.5):
        self.num_types = num_types
        self.noise_limits = noise_limits
        self.signal_frequency = signal_frequency
        self.noise_channel_prob = noise_channel_prob

    def add_white_noise(self, ecg_signal, noise_level):
        noise = torch.randn_like(ecg_signal) * noise_level
        return ecg_signal + noise

    def add_respiratory_artifact(self, ecg_signal, frequency, noise_level):
        batch_size, samples = ecg_signal.shape
        t = torch.arange(samples).float() / self.signal_frequency
        t = t.view(1, -1)  # Shape: (1, samples)
        t = t.expand(batch_size, -1)  # Shape: (batch_size, samples)
        respiratory_wave = noise_level * torch.sin(2 * np.pi * frequency * t)
        return ecg_signal + respiratory_wave

    def add_emg_artifact(self, ecg_signal, num_bursts, noise_level, burst_len_limits=(0.5, 2)):
        batch_size, signal_length = ecg_signal.shape
        noisy_signal = ecg_signal.clone()
        b, a = butter(N=5, Wn=[10, 60], btype="bandpass", fs=self.signal_frequency)
        
        for batch in range(batch_size):
            emg_signal = filtfilt(b, a, torch.randn(signal_length))
            for _ in range(num_bursts):
                burst_len = round(random.uniform(*burst_len_limits) * self.signal_frequency)
                burst_start = torch.randint(0, signal_length - burst_len, (1,)).item()
                burst_end = burst_start + burst_len
                noisy_signal[batch, burst_start:burst_end] += torch.tensor(emg_signal[burst_start:burst_end]) * noise_level
        return noisy_signal

    def add_burst_artifact(self, ecg_signal, noise_level=2):
        return ecg_signal  # Placeholder for burst artifact implementation

    def add_noise_to_channels(self, clean_signal, selected_noise_types = ['white']):
        # Create multi-channel signal from single-channel input
        batch_size, num_channels, samples = clean_signal.shape
        multi_channel = clean_signal
        noisy_multi_channel = clean_signal.clone()
        
        # Randomly select channels to add noise to
        for batch in range(batch_size):
            for channel in range(num_channels):
                if random.random() < self.noise_channel_prob:
                    # Set random total noise level
                    total_noise_level = random.uniform(*self.noise_limits) * (
                        clean_signal[batch].max() - clean_signal[batch].min()
                    ).item()
                    
                    # Select random noise types and their contributions
                    if selected_noise_types is None:
                        selected_noise_types = random.sample(self.NOISE_TYPES, self.num_types)
                    contributions = np.random.dirichlet(np.ones(self.num_types))
                    
                    channel_signal = multi_channel[batch, channel]
                    noisy_signal = channel_signal.clone()
                    
                    # Add selected noise types
                    for noise_type, contribution in zip(selected_noise_types, contributions):
                        noise_level = total_noise_level * contribution.item()
                        
                        if noise_type == 'white':
                            noisy_signal = self.add_white_noise(noisy_signal.unsqueeze(0), noise_level).squeeze(0)
                        elif noise_type == 'respiratory':
                            noisy_signal = self.add_respiratory_artifact(noisy_signal.unsqueeze(0), frequency=0.3, noise_level=noise_level).squeeze(0)
                        elif noise_type == 'emg':
                            noisy_signal = self.add_emg_artifact(noisy_signal.unsqueeze(0), num_bursts=5, noise_level=noise_level).squeeze(0)
                        elif noise_type == 'burst':
                            noisy_signal = self.add_burst_artifact(noisy_signal.unsqueeze(0)).squeeze(0)
                    
                    noisy_multi_channel[batch, channel] = noisy_signal
        
        return noisy_multi_channel


def get_losses(padded_batch_list, output, criterion, mask):

    # Calculate mask ratios for proper scaling
    total_elements = mask.numel()
    masked_elements = mask.sum()
    non_masked_elements = total_elements - masked_elements
    masked_ratio = masked_elements / total_elements
    non_masked_ratio = non_masked_elements / total_elements

    loss_full = criterion(output, padded_batch_list)
    loss = loss_full.mean()
    loss_bas = loss_full[:, :8, :].mean()
    loss_resp = loss_full[:, 8:13, :].mean()
    loss_ekg = loss_full[:, 13:14, :].mean()
    loss_emg = loss_full[:, 14:16, :].mean()

    loss_rearranged = rearrange(loss_full, 'b ch (s e) -> b ch s e', s=60, e=640)

    loss_masked = (loss_rearranged * mask.unsqueeze(-1)).mean() / masked_ratio
    loss_bas_masked = (loss_rearranged[:, :8, :] * mask.unsqueeze(-1)[:, :8, :]).mean() / masked_ratio
    loss_resp_masked = (loss_rearranged[:, 8:13, :] * mask.unsqueeze(-1)[:, 8:13, :]).mean() / masked_ratio
    loss_ekg_masked = (loss_rearranged[:, 13:14, :] * mask.unsqueeze(-1)[:, 13:14, :]).mean() / masked_ratio
    loss_emg_masked = (loss_rearranged[:, 14:16, :] * mask.unsqueeze(-1)[:, 14:16, :]).mean() / masked_ratio

    loss_non_masked = (loss_rearranged * (~mask.unsqueeze(-1)).float()).mean() / non_masked_ratio
    loss_bas_non_masked = (loss_rearranged[:, :8, :] * (~mask).unsqueeze(-1)[:, :8, :].float()).mean() / non_masked_ratio
    loss_resp_non_masked = (loss_rearranged[:, 8:13, :] * (~mask).unsqueeze(-1)[:, 8:13, :].float()).mean() / non_masked_ratio
    loss_ekg_non_masked = (loss_rearranged[:, 13:14, :] * (~mask).unsqueeze(-1)[:, 13:14, :].float()).mean() / non_masked_ratio
    loss_emg_non_masked = (loss_rearranged[:, 14:16, :] * (~mask).unsqueeze(-1)[:, 14:16, :].float()).mean() / non_masked_ratio

    return loss_full, loss, loss_bas, loss_resp, loss_ekg, loss_emg, loss_masked, loss_bas_masked, loss_resp_masked, loss_ekg_masked, loss_emg_masked, loss_non_masked, loss_bas_non_masked, loss_resp_non_masked, loss_ekg_non_masked, loss_emg_non_masked


def get_losses_fft(padded_batch_list, output, criterion, mask):

    def compute_fft_chunks(data, sample_rate=128):
        # Rearrange to (batch, channels, time_windows, samples_per_window)
        x = rearrange(data, 'b c (t w) -> b c t w', t=60, w=640)
        
        # Compute FFT for all chunks in parallel
        # Output shape: (batch, channels, time_windows, freq_bins)
        fft_result = torch.fft.rfft(x, dim=-1)
        
        # Compute amplitude and phase
        amplitude = torch.abs(fft_result)[:,:,:,:-1] / 640  # Normalize by window length
        amplitude = (torch.log10(amplitude + 1e-6) + 6)
        phase = torch.angle(fft_result)[:,:,:,:-1]
        
        # Compute frequency array (only need to do once as it's same for all chunks)
        frequencies = torch.fft.rfftfreq(640, d=1/sample_rate)[:-1]
        
        return frequencies, amplitude, phase

    def phase_loss_unreduced_sin(phase_target, phase_reconstruct):
        # Compute phase difference and divide by 2
        phase_diff = (phase_target - phase_reconstruct) / 2.0 # double the period of the phase for the loss to be maximum at pi and -pi
        
        # Compute sine squared of the difference
        # This automatically handles the circular nature of phase
        loss = torch.sin(phase_diff)**2
        
        return loss

    def phase_loss_unreduced(phase_target, phase_reconstruct):
        # Compute phase difference and divide by 2
        #phase_diff = (phase_target - phase_reconstruct) / 2.0 # double the period of the phase for the loss to be maximum at pi and -pi
        
        # Compute sine squared of the difference
        # This automatically handles the circular nature of phase
        #loss = torch.sin(phase_diff)**2

        offsets = torch.tensor([-2 * torch.pi, 0, 2 * torch.pi], device=phase_target.device)
        offsets = offsets.view(3, 1, 1, 1, 1)
        phase_target = phase_target.unsqueeze(0)
        phase_reconstruct = phase_reconstruct.unsqueeze(0)
        differences = (phase_target - phase_reconstruct + offsets) ** 2
        loss = torch.min(differences, dim=0).values
        
        return loss
    
    # Calculate mask ratios for proper scaling
    total_elements = mask.numel()
    masked_elements = mask.sum()
    non_masked_elements = total_elements - masked_elements
    masked_ratio = masked_elements / total_elements
    non_masked_ratio = non_masked_elements / total_elements
    
    freq_target, amp_target, phase_target = compute_fft_chunks(padded_batch_list)
    output_data = rearrange(output, 'b c (t w) -> b c t w', t=60, w=640)
    amp_output = output_data[..., :320]
    phase_output = output_data[..., 320:]

    #print(f'amp_target: {amp_target.shape}')
    #print(f'amp_output: {amp_output.shape}')

    full_loss_amp = criterion(amp_output, amp_target)
    full_loss_phase = phase_loss_unreduced(phase_output, phase_target)
    
    loss_amp = full_loss_amp.mean()
    loss_phase = full_loss_phase.mean()

    #print(f'full_loss_amp: {full_loss_amp}')
    #print(f'full_loss_phase: {full_loss_phase}')

    loss_amp_bas = full_loss_amp[:, :8, :].mean()
    loss_phase_bas = full_loss_phase[:, :8, :].mean()
    loss_amp_resp = full_loss_amp[:, 8:13, :].mean()
    loss_phase_resp = full_loss_phase[:, 8:13, :].mean()
    loss_amp_ekg = full_loss_amp[:, 13:14, :].mean()
    loss_phase_ekg = full_loss_phase[:, 13:14, :].mean()
    loss_amp_emg = full_loss_amp[:, 14:16, :].mean()
    loss_phase_emg = full_loss_phase[:, 14:16, :].mean()

    loss_amp_rearranged = full_loss_amp# rearrange(full_loss_amp, 'b ch (s e) -> b ch s e', s=60, e=320)
    loss_phase_rearranged = full_loss_phase# rearrange(full_loss_phase, 'b ch (s e) -> b ch s e', s=60, e=320)

    loss_amp_masked = (loss_amp_rearranged * mask.unsqueeze(-1)).mean() / masked_ratio
    loss_phase_masked = (loss_phase_rearranged * mask.unsqueeze(-1)).mean() / masked_ratio

    loss_amp_bas_masked = (loss_amp_rearranged[:, :8, :] * mask.unsqueeze(-1)[:, :8, :]).mean() / masked_ratio
    loss_phase_bas_masked = (loss_phase_rearranged[:, :8, :] * mask.unsqueeze(-1)[:, :8, :]).mean() / masked_ratio
    loss_amp_resp_masked = (loss_amp_rearranged[:, 8:13, :] * mask.unsqueeze(-1)[:, 8:13, :]).mean() / masked_ratio
    loss_phase_resp_masked = (loss_phase_rearranged[:, 8:13, :] * mask.unsqueeze(-1)[:, 8:13, :]).mean() / masked_ratio
    loss_amp_ekg_masked = (loss_amp_rearranged[:, 13:14, :] * mask.unsqueeze(-1)[:, 13:14, :]).mean() / masked_ratio
    loss_phase_ekg_masked = (loss_phase_rearranged[:, 13:14, :] * mask.unsqueeze(-1)[:, 13:14, :]).mean() / masked_ratio
    loss_amp_emg_masked = (loss_amp_rearranged[:, 14:16, :] * mask.unsqueeze(-1)[:, 14:16, :]).mean() / masked_ratio
    loss_phase_emg_masked = (loss_phase_rearranged[:, 14:16, :] * mask.unsqueeze(-1)[:, 14:16, :]).mean() / masked_ratio

    loss_amp_non_masked = (loss_amp_rearranged * (~mask.unsqueeze(-1)).float()).mean() / non_masked_ratio
    loss_phase_non_masked = (loss_phase_rearranged * (~mask.unsqueeze(-1)).float()).mean() / non_masked_ratio

    loss_amp_bas_non_masked = (loss_amp_rearranged[:, :8, :] * (~mask).unsqueeze(-1)[:, :8, :].float()).mean() / non_masked_ratio
    loss_phase_bas_non_masked = (loss_phase_rearranged[:, :8, :] * (~mask).unsqueeze(-1)[:, :8, :].float()).mean() / non_masked_ratio
    loss_amp_resp_non_masked = (loss_amp_rearranged[:, 8:13, :] * (~mask).unsqueeze(-1)[:, 8:13, :].float()).mean() / non_masked_ratio
    loss_phase_resp_non_masked = (loss_phase_rearranged[:, 8:13, :] * (~mask).unsqueeze(-1)[:, 8:13, :].float()).mean() / non_masked_ratio
    loss_amp_ekg_non_masked = (loss_amp_rearranged[:, 13:14, :] * (~mask).unsqueeze(-1)[:, 13:14, :].float()).mean() / non_masked_ratio
    loss_phase_ekg_non_masked = (loss_phase_rearranged[:, 13:14, :] * (~mask).unsqueeze(-1)[:, 13:14, :].float()).mean() / non_masked_ratio
    loss_amp_emg_non_masked = (loss_amp_rearranged[:, 14:16, :] * (~mask).unsqueeze(-1)[:, 14:16, :].float()).mean() / non_masked_ratio
    loss_phase_emg_non_masked = (loss_phase_rearranged[:, 14:16, :] * (~mask).unsqueeze(-1)[:, 14:16, :].float()).mean() / non_masked_ratio

    return full_loss_amp, full_loss_phase, loss_amp, loss_phase, loss_amp_bas, loss_phase_bas, loss_amp_resp, loss_phase_resp, loss_amp_ekg, loss_phase_ekg, loss_amp_emg, loss_phase_emg, loss_amp_masked, loss_phase_masked, loss_amp_bas_masked, loss_phase_bas_masked, loss_amp_resp_masked, loss_phase_resp_masked, loss_amp_ekg_masked, loss_phase_ekg_masked, loss_amp_emg_masked, loss_phase_emg_masked, loss_amp_non_masked, loss_phase_non_masked, loss_amp_bas_non_masked, loss_phase_bas_non_masked, loss_amp_resp_non_masked, loss_phase_resp_non_masked, loss_amp_ekg_non_masked, loss_phase_ekg_non_masked, loss_amp_emg_non_masked, loss_phase_emg_non_masked


def get_mask(x, ratio=0.34):
    device = x.device
    # Rearrange input
    x = rearrange(x, 'b ch (s e) -> b ch s e', s=60, e=640)
    
    # Create mask of shape (b, ch, s)
    mask = torch.zeros(x.shape[0], x.shape[1], x.shape[2], device=device)
    
    # Calculate number of positions to mask per sequence
    n_mask = int(mask.shape[2] * ratio)  # Only use s dimension for ratio
    
    # Create masks efficiently for all batches at once
    indices = torch.rand(x.shape[0], x.shape[1], x.shape[2], device=device).argsort(dim=-1)  # Sort along s dimension
    mask.scatter_(2, indices[:, :, :n_mask], 1)

    mask = mask.bool()
    
    return mask

def replace_masked(x, mask, value):
    # Rearrange input
    x = rearrange(x, 'b ch (s e) -> b ch s e', s=60, e=640)
    
    # Expand mask for broadcasting
    mask = mask.unsqueeze(-1).bool()
    
    # Use torch.where instead of masked_fill
    x_masked = torch.where(mask, value, x)
    
    # Rearrange back
    x_masked = rearrange(x_masked, 'b ch s e -> b ch (s e)', s=60, e=640)
    
    return x_masked


def save_data(data: Any, filename: str) -> None:
    """
    Save data to a file in either pickle, JSON, YAML, or NPY format based on the file extension.

    Parameters:
    - data: The data to save.
    - filename: The name of the file to save the data to. Should have .pickle, .pkl, .p, .json, .yaml, or .npy extension.
    """
    if filename.endswith(('.pkl', '.pickle', '.p')):
        with open(filename, 'wb') as f:
            pickle.dump(data, f)
    elif filename.endswith('.json'):
        with open(filename, 'w') as f:
            json.dump(data, f, indent=4)
    elif filename.endswith('.yaml'):
        with open(filename, 'w') as f:
            yaml.dump(data, f, default_flow_style=False)
    elif filename.endswith('.npy'):
        np.save(filename, data)
    else:
        raise ValueError("Filename must end with .pkl, .pickle, .p, .json, .yaml, or .npy")


def load_data(filename: str) -> Any:
    """
    Load data from a file in either pickle, JSON, YAML, or NPY format based on the file extension.

    Parameters:
    - filename: The name of the file to load the data from. Should have .pickle, .pkl, .p, .json, .yaml, or .npy extension.

    Returns:
    - The loaded data.
    """
    if filename.endswith(('.pkl', '.pickle', '.p')):
        with open(filename, 'rb') as f:
            return pickle.load(f)
    elif filename.endswith('.json'):
        with open(filename, 'r') as f:
            return json.load(f)
    elif filename.endswith('.yaml'):
        with open(filename, 'r') as f:
            return yaml.load(f, Loader=yaml.FullLoader)
    elif filename.endswith('.npy'):
        return np.load(filename, allow_pickle=True)
    else:
        raise ValueError("Filename must end with .pkl, .pickle, .p, .json, .yaml, or .npy")


def initialize_mae_weights(model):
    """
    Custom initialization function for MAE models with various layer types.
    """
    for name, module in model.named_modules():
        if isinstance(module, nn.Conv1d):
            # Use Kaiming Initialization for Conv1d layers (suitable for ReLU/ELU activations)
            nn.init.kaiming_uniform_(module.weight, nonlinearity='relu')
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Linear):
            # Xavier Initialization for Linear layers
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)

        elif isinstance(module, nn.BatchNorm1d):
            # Initialize BatchNorm weights and biases
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, nn.LayerNorm):
            # Default initialization for LayerNorm (PyTorch already initializes well)
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            # Initialize embedding layers (if any)
            nn.init.normal_(module.weight, mean=0, std=0.01)

        elif isinstance(module, nn.TransformerEncoderLayer):
            # Initialize Transformer components
            # Self-Attention Projection
            for sub_module in module.children():
                if isinstance(sub_module, nn.MultiheadAttention):
                    nn.init.xavier_uniform_(sub_module.in_proj_weight)
                    nn.init.xavier_uniform_(sub_module.out_proj.weight)
                    if sub_module.in_proj_bias is not None:
                        nn.init.zeros_(sub_module.in_proj_bias)
                        nn.init.zeros_(sub_module.out_proj.bias)
                
                # Linear layers in feedforward
                elif isinstance(sub_module, nn.Linear):
                    nn.init.xavier_uniform_(sub_module.weight)
                    if sub_module.bias is not None:
                        nn.init.zeros_(sub_module.bias)

                # LayerNorm (handled separately if needed)
                elif isinstance(sub_module, nn.LayerNorm):
                    nn.init.ones_(sub_module.weight)
                    nn.init.zeros_(sub_module.bias)
