"""Sleep staging and age estimation on frozen embeddings (one jointly trained head).

Model: transformer + BiLSTM over the 5-second patch embeddings of a whole recording;
per-patch 5-class staging output and a pooled age output. Loss = staging cross-entropy
(class weights 1/4/2/4/3 for W/N1/N2/N3/REM; unscored patches ignored) + age_loss_weight x
age MSE. Labels come from the StageName column (sleepbench.labels).

Config keys of note:
  selection   list of epoch-selection criteria, each evaluated on test:
              'all'/'clean' = best validation macro-F1 on all / subject-disjoint
              validation recordings; 'age'/'age-clean' = lowest validation age MAE
  early_stop  criteria that must all stall for `patience` epochs before stopping
  few_shot    {splits_path, size, rep, repeat_to}: train on one few-shot subset
  val_max_files, unknown_as_wake, age_loss_weight, sleep_loss_weight
Outputs (results_path): <run>_history.json, <run>_select-<criterion>_test.npz with
per-recording targets, probabilities and age predictions.

Usage:  python -m sleepbench.downstream.sleep_age --config <yaml> [--smoke]
"""

import argparse
import json
import math
import os
import random
import sys
import time

import h5py
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import yaml
from einops import rearrange
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

from ..labels import load_stage_labels

CLASS_WEIGHTS = [1, 4, 2, 4, 3]


def study_id(path):
    return os.path.basename(path).split('.')[0]


def few_shot_ids(fs):
    """Study IDs of one few-shot subset (file names in the splits JSON)."""
    with open(fs['splits_path']) as f:
        return {study_id(n) for n in json.load(f)[str(fs['size'])][int(fs['rep'])]}


def apply_few_shot(config, train_ds, val_ds):
    fs = config.get('few_shot')
    if fs:
        ids = few_shot_ids(fs)
        files = [f for f in train_ds.files if study_id(f) in ids]
        reps = max(int(fs.get('repeat_to', 1024)) // max(len(files), 1), 1)
        train_ds.files = files * reps
        print(f'few-shot size {fs["size"]} rep {fs["rep"]}: {len(files)} of {len(ids)} subset '
              f'recordings eligible, x{reps} = {len(train_ds.files)} items per epoch', flush=True)
    if config.get('val_max_files'):
        val_ds.files = val_ds.files[:config['val_max_files']]
        print(f'validation limited to {len(val_ds.files)} recordings', flush=True)


def load_split_ids(split_path, split):
    with open(split_path) as f:
        return [study_id(p) for p in json.load(f)[split]]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class SleepAgeDataset(Dataset):
    """Frozen embeddings + sleep stage labels + age, one item per recording."""

    def __init__(self, config, split, pretrain_type, unknown_as_wake=False):
        self.unknown_as_wake = unknown_as_wake

        demo = pd.read_csv(config['demographics_labels_path'])
        self.age_map = dict(zip(demo['Study ID'].astype(str),
                                demo['Age at Study Date'].astype(float) / 100.0))

        label_dir = os.path.join(config['labels_path'], config['dataset'])
        self.label_map = {study_id(f): os.path.join(label_dir, f)
                          for f in os.listdir(label_dir) if f.endswith('.csv')}

        # Eligible: in split, raw PSG present, label file and age present
        raw = set(study_id(f) for f in os.listdir(config['raw_data_dir']))
        ids = set(load_split_ids(config['split_path'], split))
        ids &= raw & set(self.label_map) & set(self.age_map)

        emb_root = os.path.join(config['embedding_path'], pretrain_type)
        emb = []
        for d, _, files in os.walk(emb_root):
            emb += [os.path.join(d, f) for f in files if f.endswith('.hdf5')]
        self.files = sorted(f for f in emb if study_id(f) in ids)
        if config.get('max_files'):
            self.files = self.files[:config['max_files']]
        print(f'{split}: {len(self.files)} recordings', flush=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        path = self.files[idx]
        sid = study_id(path)
        y = load_stage_labels(self.label_map[sid])
        if self.unknown_as_wake:
            y = np.where(y < 0, 0, y)
        with h5py.File(path, 'r') as hf:
            x = np.stack([hf[k][:] for k in hf.keys()])
        x = torch.tensor(x, dtype=torch.float32)
        if x.ndim == 3:
            x = x.mean(dim=0)
        n = min(x.shape[0], len(y))
        return x[:n], torch.tensor(y[:n], dtype=torch.long), \
            torch.tensor(self.age_map[sid], dtype=torch.float32), sid


def make_collate(max_len):
    def collate(batch):
        xs, ys, ages, sids = zip(*batch)
        full_len = [len(y) for y in ys]
        T = min(max(full_len), max_len)
        B, D = len(xs), xs[0].shape[-1]
        x = torch.zeros(B, T, D)
        y = torch.full((B, T), -1, dtype=torch.long)
        pad = torch.ones(B, T, dtype=torch.bool)  # True = padding
        for i, (xi, yi) in enumerate(zip(xs, ys)):
            s = min(len(yi), T)
            x[i, :s], y[i, :s], pad[i, :s] = xi[:s], yi[:s], False
        return x, y, torch.stack(ages), pad, list(sids), full_len
    return collate


# ---------------------------------------------------------------------------
# Model (architecture identical to fine_tune_sleep_age.py)
# ---------------------------------------------------------------------------

class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, d_model):
        super().__init__()
        pos = torch.arange(max_seq_len).unsqueeze(1)
        div = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
        pe = torch.zeros(max_seq_len, d_model)
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        return x + self.pe[:, :x.size(1), :]


class AttentionPooling(nn.Module):
    def __init__(self, d, num_heads, dropout):
        super().__init__()
        self.layer = nn.TransformerEncoderLayer(d_model=d, nhead=num_heads, dropout=dropout,
                                                batch_first=True)

    def forward(self, x, pad):
        h = self.layer(x, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).to(h.dtype)
        return (h * keep).sum(1) / keep.sum(1).clamp(min=1)


class SleepAgeLSTMClassifier(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, num_classes,
                 pooling_head=4, dropout=0.1, max_seq_length=8196):
        super().__init__()
        self.pos_enc = PositionalEncoding(max_seq_length, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        enc_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads,
                                               dropout=dropout, batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc_layer, num_layers=num_layers)
        lstm_drop = dropout if num_layers > 1 else 0.0
        self.lstm = nn.LSTM(embed_dim, embed_dim // 2, num_layers=num_layers,
                            batch_first=True, dropout=lstm_drop, bidirectional=True)
        self.fc_sleep = nn.Linear(embed_dim, num_classes)
        self.temporal_pool = AttentionPooling(embed_dim, pooling_head, dropout)
        self.fc_age = nn.Sequential(nn.Linear(embed_dim, 1), nn.Softplus())

    def forward(self, x, pad):
        T = x.size(1)
        x = self.norm(self.pos_enc(x))
        x = self.transformer(x, src_key_padding_mask=pad)
        lengths = (~pad).sum(1).cpu()
        packed = pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        x, _ = self.lstm(packed)
        x, _ = pad_packed_sequence(x, batch_first=True, total_length=T)
        return self.fc_sleep(x), self.fc_age(self.temporal_pool(x, pad)).squeeze(-1)


def masked_ce_loss(logits, y, pad):
    valid = ~pad & (y >= 0)
    w = torch.tensor(CLASS_WEIGHTS, dtype=torch.float32, device=logits.device)
    return F.cross_entropy(logits[valid].float(), y[valid], weight=w)


def selection_score(rec, criterion):
    """Higher is better: macro-F1, or negative age MAE."""
    subset = 'val_clean' if criterion.endswith('clean') else 'val_all'
    if criterion.startswith('age'):
        return -rec[subset]['age_mae']
    return rec[subset]['macro_f1']


def f1_from_counts(tp, fp, fn):
    p = tp / (tp + fp + 1e-7)
    r = tp / (tp + fn + 1e-7)
    return 2 * p * r / (p + r + 1e-7)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def run_inference(model, loader, device):
    """Per-recording outputs: {sid: (targets, probs, age_pred, age_true, full_len)}."""
    model.eval()
    out = {}
    for x, y, age, pad, sids, full_len in loader:
        x, pad = x.to(device), pad.to(device)
        with torch.autocast('cuda', enabled=device.type == 'cuda'):
            logits, age_pred = model(x, pad)
        probs = F.softmax(logits.float(), dim=-1).cpu().numpy()
        age_pred = age_pred.float().cpu().numpy()
        for i, sid in enumerate(sids):
            n = int((~pad[i]).sum())
            out[sid] = (y[i, :n].numpy(), probs[i, :n], float(age_pred[i]),
                        float(age[i]), full_len[i])
    return out


def summarise(out, sids, C=5):
    tp, fp, fn = np.zeros(C), np.zeros(C), np.zeros(C)
    ae = []
    for s in sids:
        y, p, a_pred, a_true, _ = out[s]
        v = y >= 0
        t, q = y[v], p[v].argmax(1)
        for c in range(C):
            tp[c] += ((q == c) & (t == c)).sum()
            fp[c] += ((q == c) & (t != c)).sum()
            fn[c] += ((q != c) & (t == c)).sum()
        ae.append(abs(a_pred - a_true) * 100)
    f1 = f1_from_counts(tp, fp, fn)
    return {'macro_f1': float(f1.mean()), 'per_class_f1': f1.round(4).tolist(),
            'accuracy': float(tp.sum() / max(tp.sum() + fp.sum(), 1)),
            'age_mae': float(np.mean(ae)) if ae else float('nan'), 'n': len(sids)}


def save_test(out, path, meta):
    sids = sorted(out)
    lens = [len(out[s][0]) for s in sids]
    np.savez_compressed(
        path,
        study_id=np.array(sids),
        offsets=np.concatenate([[0], np.cumsum(lens)]).astype(np.int64),
        targets=np.concatenate([out[s][0] for s in sids]).astype(np.int8),
        probs=np.concatenate([out[s][1] for s in sids]).astype(np.float16),
        age_pred=np.array([out[s][2] * 100 for s in sids]),
        age_true=np.array([out[s][3] * 100 for s in sids]),
        full_len=np.array([out[s][4] for s in sids]),
        meta=json.dumps(meta))
    print(f'saved {path}', flush=True)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', required=True)
    ap.add_argument('--smoke', action='store_true',
                    help='tiny run: 40/20/20 recordings, 2 epochs, test batch 1 vs default compared')
    args = ap.parse_args()
    config = yaml.safe_load(open(args.config))
    if args.smoke:
        config.update(max_files=40, epochs=2, run_name=config['run_name'] + '_smoke')

    seed = config.get('seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mp = config['model_params']
    out_dir = config['results_path']
    os.makedirs(out_dir, exist_ok=True)
    run = config['run_name']
    w_age = float(config['age_loss_weight'])
    w_sleep = float(config.get('sleep_loss_weight', 1.0))
    pretrain_type = config['pretrain_type']
    collate = make_collate(mp['max_seq_length'])
    print(f'run={run} pretrain_type={pretrain_type} age_loss_weight={w_age} sleep_loss_weight={w_sleep} '
          f'unknown_as_wake={config.get("unknown_as_wake", False)} device={device}', flush=True)

    train_ds = SleepAgeDataset(config, 'pretrain', pretrain_type,
                               unknown_as_wake=config.get('unknown_as_wake', False))
    val_ds = SleepAgeDataset(config, 'validation', pretrain_type)
    apply_few_shot(config, train_ds, val_ds)
    nw = config['num_workers']
    train_dl = DataLoader(train_ds, batch_size=config['batch_size'], shuffle=True,
                          num_workers=nw, collate_fn=collate)
    val_dl = DataLoader(val_ds, batch_size=max(1, config['batch_size'] // 4), shuffle=False,
                        num_workers=nw, collate_fn=collate)

    val_ids = [study_id(f) for f in val_ds.files]
    clean = None
    if config.get('val_clean_ids_path'):
        clean_set = set(open(config['val_clean_ids_path']).read().split())
        clean = [s for s in val_ids if s in clean_set]
        print(f'validation: {len(clean)} subject-disjoint / {len(val_ids)}', flush=True)

    model = SleepAgeLSTMClassifier(mp['embed_dim'], mp['num_heads'], mp['num_layers'],
                                   mp['num_classes'], mp['pooling_head'], mp['dropout'],
                                   mp['max_seq_length']).to(device)
    opt = optim.AdamW(model.parameters(), lr=config['lr'])
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    crit = [c for c in config.get('selection', ['all', 'clean'])
            if clean is not None or not c.endswith('clean')]
    best = {c: {'score': -float('inf'), 'epoch': -1, 'state': None} for c in crit}
    stop_on = config.get('early_stop', crit[:1])
    print(f'selection criteria: {crit} (early stopping when {stop_on} all stall)', flush=True)
    history, patience_ctr = [], 0
    for epoch in range(config['epochs']):
        t0 = time.time()
        model.train()
        tot, nb = 0.0, 0
        for x, y, age, pad, _, _ in train_dl:
            x, y, age, pad = x.to(device), y.to(device), age.to(device), pad.to(device)
            opt.zero_grad()
            with torch.autocast('cuda', enabled=device.type == 'cuda'):
                logits, age_pred = model(x, pad)
                loss = 0.0
                if w_sleep > 0:
                    loss = loss + w_sleep * masked_ce_loss(logits, y, pad)
                if w_age > 0:
                    loss = loss + w_age * F.mse_loss(age_pred.float(), age)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            tot += loss.item(); nb += 1

        val_out = run_inference(model, val_dl, device)
        rec = {'epoch': epoch, 'train_loss': tot / max(nb, 1), 'secs': round(time.time() - t0),
               'gpu_peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2)
               if device.type == 'cuda' else 0.0,
               'val_all': summarise(val_out, val_ids)}
        if clean is not None:
            rec['val_clean'] = summarise(val_out, clean)
            rec['val_overlap'] = summarise(val_out, sorted(set(val_ids) - set(clean)))
        history.append(rec)
        print(json.dumps({k: (v if not isinstance(v, dict) else
                              {kk: v[kk] for kk in ('macro_f1', 'age_mae')})
                          for k, v in rec.items()}), flush=True)

        for c in crit:
            score = selection_score(rec, c)
            if score > best[c]['score']:
                best[c] = {'score': score, 'epoch': epoch,
                           'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        patience_ctr = 0 if any(best[c]['epoch'] == epoch for c in stop_on) else patience_ctr + 1
        if patience_ctr >= config['patience']:
            print(f'early stopping at epoch {epoch}', flush=True)
            break

    json.dump({'config': config, 'history': history,
               'best_epoch': {c: best[c]['epoch'] for c in crit}},
              open(os.path.join(out_dir, f'{run}_history.json'), 'w'), indent=1)

    test_ds = SleepAgeDataset(config, 'test', pretrain_type)
    test_bs = config.get('test_batch_size', max(1, config['batch_size'] // 2))
    test_dl = DataLoader(test_ds, batch_size=test_bs, shuffle=False, num_workers=nw,
                         collate_fn=collate)
    tested = {}
    for c in crit:
        if best[c]['epoch'] in tested:
            print(f'{c} selection picks the same epoch as {tested[best[c]["epoch"]]}; '
                  f'no extra test pass', flush=True)
            continue
        tested[best[c]['epoch']] = c
        model.load_state_dict(best[c]['state'])
        torch.save(best[c]['state'], os.path.join(out_dir, f'{run}_select-{c}_model.pt'))
        out = run_inference(model, test_dl, device)
        res = summarise(out, sorted(out))
        print(f'TEST select-{c} (epoch {best[c]["epoch"]}): {json.dumps(res)}', flush=True)
        save_test(out, os.path.join(out_dir, f'{run}_select-{c}_test.npz'),
                  {'run': run, 'select': c, 'epoch': best[c]['epoch'], 'token_level': res,
                   'n_truncated': int(sum(o[4] > mp['max_seq_length'] for o in out.values()))})

        if args.smoke and c == crit[0]:
            dl1 = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=nw, collate_fn=collate)
            out1 = run_inference(model, dl1, device)
            dp = max(np.abs(out[s][1] - out1[s][1]).max() for s in out)
            da = max(abs(out[s][2] - out1[s][2]) * 100 for s in out)
            print(f'SMOKE batch-invariance: max|dprob|={dp:.2e} max|dage|={da:.2e} years', flush=True)


if __name__ == '__main__':
    main()
