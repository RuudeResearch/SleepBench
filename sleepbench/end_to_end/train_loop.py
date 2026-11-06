"""Shared training/validation loop and model factory for the E2E baseline."""

import math
import os

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import tqdm

from .dataset import RawSignalDataset, make_collate_fn
from .model import EndToEndSleepModel
from .utils import (
    cox_ph_loss,
    load_json,
    load_yaml,
    masked_cross_entropy_loss,
    stage_valid_mask,
)


# Which losses to include in the backwards pass. Overridden per task.
TASK_LOSS_WEIGHTS = {
    "sleep_age": {"sleep": 1.0, "age": 10.0, "diagnosis": 0.0, "death": 0.0, "ahi": 0.0},
    "ahi": {"sleep": 0.0, "age": 0.0, "diagnosis": 0.0, "death": 0.0, "ahi": 1.0},
    "death_diagnosis": {"sleep": 0.0, "age": 0.0, "diagnosis": 1.0, "death": 1.0, "ahi": 0.0},
}


def build_model(config, device):
    model = EndToEndSleepModel(
        input_size=config["input_size"],
        encoder_output_size=config["encoder_output_size"],
        encoder_num_heads=config["encoder_num_heads"],
        encoder_num_layers=config["encoder_num_layers"],
        tokens_per_chunk=config["tokens_per_chunk"],
        encoder_dropout=config["encoder_dropout"],
        head_num_heads=config["model_params"]["num_heads"],
        head_num_layers=config["model_params"]["num_layers"],
        num_classes=config["model_params"]["num_classes"],
        pooling_head=config["model_params"]["pooling_head"],
        head_dropout=config["model_params"]["dropout"],
        max_seq_length=config["model_params"]["max_seq_length"],
        num_diagnosis=config["num_diagnosis"],
        encoder_chunks_per_pass=config["encoder_chunks_per_pass"],
    )
    return model.to(device)


def build_loaders(config, channel_groups, include_ahi, task=None):
    kw = dict(include_ahi=include_ahi, task=task)
    train_ds = RawSignalDataset(config, channel_groups, split="pretrain", **kw)
    val_ds = RawSignalDataset(config, channel_groups, split="validation", **kw)
    test_ds = RawSignalDataset(config, channel_groups, split="test", **kw)

    collate = make_collate_fn(
        tokens_per_chunk=config["tokens_per_chunk"],
        chunk_samples=config["tokens_per_chunk"] * config["input_size"],
        max_chunks=config.get("max_chunks_per_recording") or None,
    )

    common = dict(
        num_workers=config["num_workers"],
        collate_fn=collate,
        pin_memory=False,
    )

    train_loader = DataLoader(
        train_ds, batch_size=config["batch_size"], shuffle=True, drop_last=True, **common
    )
    val_loader = DataLoader(
        val_ds, batch_size=max(1, config["batch_size"] // 2), shuffle=False, drop_last=False, **common
    )
    test_loader = DataLoader(
        test_ds, batch_size=max(1, config["batch_size"] // 2), shuffle=False, drop_last=False, **common
    )
    return train_loader, val_loader, test_loader


def _compute_losses(model_out, batch, config, weights, device):
    sleep_stage, token_mask, age_out, hazards_diag, hazards_death, ahi_logit = model_out

    y_stage = batch["y_stage"].to(device)
    diagnosis_presence = batch["diagnosis_presence"].to(device)
    diagnosis_time = batch["diagnosis_time"].to(device)
    death_presence = batch["death_presence"].to(device)
    death_time = batch["death_time"].to(device)
    age_target = batch["age"].to(device)
    ahi_target = batch["ahi"].to(device)

    losses = {}
    if weights["sleep"] > 0:
        valid = stage_valid_mask(token_mask, y_stage)
        losses["sleep"] = masked_cross_entropy_loss(sleep_stage, y_stage, valid)
    else:
        losses["sleep"] = torch.zeros((), device=device)

    if weights["age"] > 0:
        losses["age"] = F.mse_loss(age_target.float(), age_out.float())
    else:
        losses["age"] = torch.zeros((), device=device)

    if weights["diagnosis"] > 0:
        losses["diagnosis"] = cox_ph_loss(hazards_diag, diagnosis_time, diagnosis_presence)
    else:
        losses["diagnosis"] = torch.zeros((), device=device)

    if weights["death"] > 0:
        losses["death"] = cox_ph_loss(hazards_death, death_time, death_presence)
    else:
        losses["death"] = torch.zeros((), device=device)

    if weights["ahi"] > 0:
        losses["ahi"] = F.binary_cross_entropy_with_logits(ahi_logit, ahi_target.float())
    else:
        losses["ahi"] = torch.zeros((), device=device)

    total = sum(weights[k] * losses[k] for k in losses)
    return total, losses


def run_iteration(model, batch, optimizer, scaler, config, device, task, mode, it=0, n_it=1):
    is_training = mode == "train"
    weights = TASK_LOSS_WEIGHTS[task]
    # optional gradient accumulation over config['accumulation_steps'] batches (1 = off,
    # the setting used for the reported results)
    accum = max(int(config.get("accumulation_steps") or 1), 1)

    x_raw = batch["x_raw"].to(device, non_blocking=True)
    chunk_mask = batch["chunk_mask"].to(device, non_blocking=True)

    if is_training and it % accum == 0:
        optimizer.zero_grad()

    cm = torch.cuda.amp.autocast if is_training else _no_autocast
    with cm() if is_training else torch.no_grad():
        out = model(x_raw, chunk_mask=chunk_mask)
        total_loss, losses = _compute_losses(out, batch, config, weights, device)

    if is_training:
        scaler.scale(total_loss / accum).backward()
        if (it + 1) % accum == 0 or it + 1 == n_it:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()

    # Per-task monitoring. Only compute the metrics relevant to the task
    # being trained so the progress bar isn't noisy with unrelated numbers.
    metrics = {f"loss_{k}": v.item() for k, v in losses.items()}
    metrics["loss"] = total_loss.item()

    with torch.no_grad():
        if weights["sleep"] > 0:
            sleep_stage = out[0]
            token_mask = out[1]
            y_stage = batch["y_stage"].to(device)
            valid = stage_valid_mask(token_mask, y_stage)
            if valid.any():
                logits = sleep_stage.reshape(-1, sleep_stage.size(-1))[valid]
                tgt = y_stage.reshape(-1).long()[valid]
                metrics["correct"] = (logits.argmax(dim=1) == tgt).sum().item()
                metrics["total"] = tgt.numel()
            else:
                metrics["correct"] = 0
                metrics["total"] = 0
        if weights["ahi"] > 0:
            ahi_pred = (torch.sigmoid(out[5]) > 0.5).float()
            metrics["ahi_correct"] = (ahi_pred == batch["ahi"].to(device)).sum().item()
            metrics["ahi_total"] = ahi_pred.numel()
        # For Cox losses we report the number of events actually seen in the
        # batch; if it's zero the loss had no gradient signal that step.
        if weights["death"] > 0:
            metrics["death_events"] = int(batch["death_presence"].sum().item())
        if weights["diagnosis"] > 0:
            metrics["diag_events"] = int(batch["diagnosis_presence"].sum().item())

    if any(isinstance(v, float) and math.isnan(v) for v in metrics.values()):
        raise ValueError(f"NaN encountered in metrics: {metrics}")

    return metrics


class _no_autocast:
    """no-op context manager used in eval mode."""

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _epoch_loop(model, loader, optimizer, scaler, config, device, task, mode, desc):
    if mode == "train":
        model.train()
    else:
        model.eval()

    weights = TASK_LOSS_WEIGHTS[task]
    running = {
        "loss": 0.0, "loss_sleep": 0.0, "loss_age": 0.0,
        "loss_diagnosis": 0.0, "loss_death": 0.0, "loss_ahi": 0.0,
        "correct": 0, "total": 0,
        "ahi_correct": 0, "ahi_total": 0,
        "death_events": 0, "diag_events": 0,
    }

    loop = tqdm(enumerate(loader), total=len(loader), desc=desc, leave=True, ncols=200)
    for i, batch in loop:
        m = run_iteration(model, batch, optimizer, scaler, config, device, task, mode,
                          it=i, n_it=len(loader))
        for k in running:
            running[k] += m.get(k, 0)
        avg = running["loss"] / (i + 1)

        # Build the postfix from the task's active losses only.
        postfix = {"loss": f"avg:{avg:.3f}"}
        if weights["sleep"] > 0 and running["total"]:
            postfix["sleep_acc"] = f"{running['correct'] / running['total']:.3f}"
        if weights["age"] > 0:
            postfix["age_l"] = f"{running['loss_age'] / (i+1):.3f}"
        if weights["ahi"] > 0 and running["ahi_total"]:
            postfix["ahi_acc"] = f"{running['ahi_correct'] / running['ahi_total']:.3f}"
        if weights["death"] > 0:
            postfix["death_l"] = f"{running['loss_death'] / (i+1):.3f}"
            postfix["death_ev"] = f"{running['death_events']}"
        if weights["diagnosis"] > 0:
            postfix["diag_l"] = f"{running['loss_diagnosis'] / (i+1):.3f}"
            postfix["diag_ev"] = f"{running['diag_events']}"
        loop.set_postfix(postfix)
    n = max(1, len(loader))
    return {k: v / n if k.startswith("loss") else v for k, v in running.items()}


def train(model, train_loader, val_loader, config, device, task, patience):
    optimizer = optim.AdamW(model.parameters(), lr=config["lr"])
    scaler = torch.cuda.amp.GradScaler()

    best_val = float("inf")
    best_state = None
    patience_counter = 0

    for epoch in range(config["epochs"]):
        _epoch_loop(
            model, train_loader, optimizer, scaler, config, device,
            task=task, mode="train", desc=f"Epoch {epoch}/{config['epochs'] - 1}",
        )
        val_metrics = _epoch_loop(
            model, val_loader, None, None, config, device,
            task=task, mode="val", desc=f"Validation {epoch}/{config['epochs'] - 1}",
        )
        val_loss = val_metrics["loss"]
        if val_loss < best_val:
            best_val = val_loss
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            patience_counter = 0
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f"Early stopping at epoch {epoch + 1} (best val loss {best_val:.4f})")
                break

        if torch.cuda.is_available():  # report peak GPU memory
            print(f"Epoch {epoch}: gpu_peak_gb={torch.cuda.max_memory_allocated() / 2**30:.2f}")
        print(f"Epoch {epoch}: val_loss={val_loss:.4f}  best={best_val:.4f}  "
              f"patience={patience_counter}/{patience}")

    if best_state is not None:
        model.load_state_dict(best_state)
    return model, best_val, optimizer, scaler


def save_checkpoint(model, optimizer, scaler, config, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scaler_state_dict": scaler.state_dict(),
        "config": config,
    }, path)
    print(f"Saved checkpoint -> {path}")


# Filename each existing evaluation notebook expects, keyed by task.
TASK_RESULT_FILENAME = {
    "sleep_age": "sleep_stage_and_age_results.npy",
    "ahi": "ahi_diagnosis_results.npy",
    "death_diagnosis": "death_and_diagnosis_results.npy",
}


def task_output_path(config, task):
    """Match the layout that pretrain_comparison/evaluation/*.ipynb expects:
    <save_path>/<run_name>/<task_filename>.npy
    """
    run_name = config.get("run_name", "end_to_end")
    return os.path.join(config["save_path"], run_name, TASK_RESULT_FILENAME[task])


def task_checkpoint_path(config, task):
    run_name = config.get("run_name", "end_to_end")
    # Keep the checkpoint name aligned with the task .npy.
    stem = TASK_RESULT_FILENAME[task].replace("_results.npy", "_model.pt")
    return os.path.join(config["save_path"], run_name, stem)


def evaluate_and_save(model, loader, output_path, device, task, config):
    """Run the model on `loader` and save predictions/targets in the format
    expected by `pretrain_comparison/evaluation/results_*.ipynb`.
    """
    model.eval()
    sleep_preds, sleep_targets = [], []      # argmax, kept for backward compat
    sleep_probs = []                          # softmax probabilities — needed for proper AUROC
    sleep_recording_ids = []                  # per-token recording index — needed for bootstrap-by-recording
    study_ids = []                            # study ID of each recording (recording index order)
    age_preds, age_targets = [], []
    ahi_preds, ahi_targets = [], []
    death_hazards, death_events, death_times = [], [], []
    diag_hazards, diag_events, diag_times = [], [], []

    with torch.no_grad():
        global_offset = 0  # rolling recording index across batches
        for batch in tqdm(loader, desc=f"Evaluating ({task})", ncols=120):
            x_raw = batch["x_raw"].to(device)
            chunk_mask = batch["chunk_mask"].to(device)
            sleep_stage, token_mask, age_out, hazards_diag, hazards_death, ahi_logit = model(
                x_raw, chunk_mask=chunk_mask,
            )
            B, S, _ = sleep_stage.shape

            y_stage = batch["y_stage"].to(device)
            valid = stage_valid_mask(token_mask, y_stage)
            if valid.any():
                logits = sleep_stage.reshape(-1, sleep_stage.size(-1))[valid]
                probs = torch.softmax(logits.float(), dim=-1)
                tgt = y_stage.reshape(-1).long()[valid]
                sleep_preds.extend(logits.argmax(dim=1).cpu().tolist())
                sleep_targets.extend(tgt.cpu().tolist())
                sleep_probs.append(probs.cpu().numpy())

                # Recording index for each kept token (for recording-level bootstrap).
                rec_per_token = (
                    torch.arange(B, device=device).unsqueeze(1).expand(B, S).reshape(-1)
                )
                rec_per_token = rec_per_token + global_offset
                sleep_recording_ids.append(rec_per_token[valid].cpu().numpy())
            global_offset += B
            study_ids.extend(os.path.basename(p).split(".")[0] for p in batch["hdf5_path"])

            age_preds.extend(age_out.cpu().numpy().flatten().tolist())
            age_targets.extend(batch["age"].cpu().numpy().flatten().tolist())

            ahi_preds.extend(torch.sigmoid(ahi_logit).cpu().numpy().flatten().tolist())
            ahi_targets.extend(batch["ahi"].cpu().numpy().flatten().tolist())

            death_hazards.append(hazards_death.cpu().numpy())
            death_events.append(batch["death_presence"].cpu().numpy())
            death_times.append(batch["death_time"].cpu().numpy())

            diag_hazards.append(hazards_diag.cpu().numpy())
            diag_events.append(batch["diagnosis_presence"].cpu().numpy())
            diag_times.append(batch["diagnosis_time"].cpu().numpy())

    death_pack = {
        "times": np.concatenate(death_times, axis=0) if death_times else np.array([]),
        "presence": np.concatenate(death_events, axis=0) if death_events else np.array([]),
        "hazards": np.concatenate(death_hazards, axis=0) if death_hazards else np.array([]),
    }
    diag_pack = {
        "times": np.concatenate(diag_times, axis=0) if diag_times else np.array([]),
        "presence": np.concatenate(diag_events, axis=0) if diag_events else np.array([]),
        "hazards": np.concatenate(diag_hazards, axis=0) if diag_hazards else np.array([]),
    }

    sleep_probs_arr = (
        np.concatenate(sleep_probs, axis=0) if sleep_probs else np.array([])
    )
    sleep_rec_ids_arr = (
        np.concatenate(sleep_recording_ids, axis=0) if sleep_recording_ids else np.array([])
    )

    results = {
        "task": task,
        "study_ids": study_ids,
        # Sleep staging: keep argmax (for backward compat) + add probs + recording IDs
        "sleep_predictions": sleep_preds,
        "sleep_targets": sleep_targets,
        "sleep_probabilities": sleep_probs_arr,
        "sleep_recording_ids": sleep_rec_ids_arr,
        "age_predictions": age_preds,
        "age_targets": age_targets,
        # Keys consumed by results_ahi.ipynb
        "ahi_diagnosis_predictions": ahi_preds,
        "ahi_diagnosis_targets": ahi_targets,
        # Nested layout consumed by results_death_and_disease.ipynb
        "death": death_pack,
        "diagnosis": diag_pack,
    }
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    np.save(output_path, results)
    print(f"Saved results -> {output_path}")
    return results


def _apply_overrides(config, overrides):
    """Shallow-merge `overrides` into `config` (drops None values)."""
    if not overrides:
        return config
    applied = {}
    for k, v in overrides.items():
        if v is None:
            continue
        config[k] = v
        applied[k] = v
    if applied:
        print(f"Config overrides applied: {applied}")
    return config


def setup_training(config_path, task, overrides=None):
    """Load the config, data loaders and model for one task. `overrides` is a flat dict
    of {key: value} applied on top of the config (None values are ignored)."""
    config = load_yaml(config_path)
    config = _apply_overrides(config, overrides)
    channel_groups = load_json(config["channel_groups_path"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    torch.manual_seed(config["seed"])
    np.random.seed(config["seed"])

    # each task uses every recording with the labels it needs (as the frozen
    # pipeline does) instead of the complete-case subset across all tasks
    include_ahi = task == "ahi"
    train_loader, val_loader, test_loader = build_loaders(config, channel_groups, include_ahi, task)

    model = build_model(config, device)
    return config, device, model, train_loader, val_loader, test_loader
