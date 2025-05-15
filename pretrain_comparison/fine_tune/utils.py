
import torch
import os
import sys
sys.path.append('/oak/stanford/groups/jamesz/magnusrk/pretraining_comparison')
from comparison.utils import *
import torch.nn.functional as F
import torch.nn as nn



def cox_ph_loss(hazards, event_times, is_event):
    hazards = hazards  # Ensures float32
    event_times = event_times
    is_event = is_event
    # Sort event times and get corresponding indices for sorting other tensors
    event_times, sorted_idx = event_times.sort(dim=0, descending=True)
    hazards = hazards.gather(0, sorted_idx)
    is_event = is_event.gather(0, sorted_idx)

    # Compute log cumulative hazard across all labels in one step
    # with autocast(enabled=False):  # Disable autocast for this block to ensure float32 precision
    #     log_cumulative_hazard = torch.logcumsumexp(hazards.float(), dim=0)

    log_cumulative_hazard = torch.logcumsumexp(hazards.float(), dim=0)

    # Calculate losses for all labels simultaneously
    losses = (hazards - log_cumulative_hazard) * is_event
    losses = -losses  # Negative for maximization

    # Average loss per label
    label_loss = losses.sum(dim=0) / (is_event.sum(dim=0) + 1e-9)  # Avoid division by zero

    # Average across labels
    total_loss = label_loss.mean()

    return total_loss

def masked_cross_entropy_loss(outputs, y_data, valid_mask):
    # Reshape outputs and labels to (B * seq_len, num_classes) and (B * seq_len,)
    B, seq_len, num_classes = outputs.shape
    #outputs = outputs.reshape(B * seq_len, num_classes)
    #y_data = y_data.reshape(B * seq_len).long()  # Convert y_data to Long for cross_entropy
    #y_data = y_data.replace(-1,0)
    #y_data = torch.where(y_data == -1, torch.tensor(0, device=y_data.device, dtype=y_data.dtype), y_data)

    output_reshaped = rearrange(outputs, 'b s c -> (b s) c')
    targets_reshaped = rearrange(y_data, 'b s -> (b s)').long()

    class_weights = {0: 1,
                    1: 4,
                    2: 2,
                    3: 4,
                    4: 3
                    }

    weights_tensor = torch.ones(num_classes, device=outputs.device)
    for cls, weight in class_weights.items():
        weights_tensor[cls] = weight

    # Verify shapes match
    
    # Only select valid samples
    #output_reshaped = output_reshaped[valid_mask]
    #targets_reshaped = targets_reshaped[valid_mask]
    # Create loss function with ignore_index=-1
    if any(valid_mask):
        output_reshaped = output_reshaped[valid_mask]
        targets_reshaped = targets_reshaped[valid_mask]

    loss = F.cross_entropy(output_reshaped, targets_reshaped, weight=weights_tensor, reduction='none')
    #if nan in loss print unique from targets_reshaped
    if torch.isnan(loss).any():
        print(torch.unique(targets_reshaped))

    # Calculate cross-entropy loss without reduction
    # loss = F.cross_entropy(outputs, y_data, reduction='none')

    # Mask out the padded elements (where mask == 1, indicating padding)
    #loss = loss * (mask == 0).float()  # Keep only real data points, ignore padding
    #loss = loss[valid_mask]

    # Average only over valid (non-padded) elements
    loss = loss.sum() / valid_mask.float().sum()
    
    return loss