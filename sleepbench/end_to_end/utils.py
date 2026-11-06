"""Shared utilities for end-to-end training (loss functions, config loading)."""

import json
import os

import torch
import torch.nn.functional as F
import yaml
from einops import rearrange


SLEEP_STAGE_CLASS_WEIGHTS = {0: 1.0, 1: 4.0, 2: 2.0, 3: 4.0, 4: 3.0}


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def cox_ph_loss(hazards, event_times, is_event):
    """Cox partial-likelihood loss (as in the downstream heads)."""
    event_times, sorted_idx = event_times.sort(dim=0, descending=True)
    hazards = hazards.gather(0, sorted_idx)
    is_event = is_event.gather(0, sorted_idx)

    log_cumulative_hazard = torch.logcumsumexp(hazards.float(), dim=0)
    losses = -((hazards - log_cumulative_hazard) * is_event)
    label_loss = losses.sum(dim=0) / (is_event.sum(dim=0) + 1e-9)
    return label_loss.mean()


def masked_cross_entropy_loss(outputs, y_data, valid_mask):
    """Per-token weighted cross-entropy over the unmasked positions."""
    _, _, num_classes = outputs.shape
    output_reshaped = rearrange(outputs, "b s c -> (b s) c")
    targets_reshaped = rearrange(y_data, "b s -> (b s)").long()

    weights = torch.ones(num_classes, device=outputs.device)
    for cls, w in SLEEP_STAGE_CLASS_WEIGHTS.items():
        if cls < num_classes:
            weights[cls] = w

    if not valid_mask.any():
        return torch.tensor(0.0, device=outputs.device, requires_grad=True)

    output_reshaped = output_reshaped[valid_mask]
    targets_reshaped = targets_reshaped[valid_mask]

    loss = F.cross_entropy(
        output_reshaped, targets_reshaped, weight=weights, reduction="none"
    )
    return loss.sum() / valid_mask.float().sum()


def stage_valid_mask(token_mask, targets):
    """True for tokens that should contribute to the sleep-stage loss."""
    mask_flat = token_mask.reshape(-1)
    target_flat = targets.reshape(-1).long()
    return (~mask_flat) & (target_flat != -1)


def ensure_dir(path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
