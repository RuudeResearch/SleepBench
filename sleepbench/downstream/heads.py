"""Recording-level heads on frozen embeddings.

--task ahi        binary AHI >= 15 (BCE)
--task diagnosis  Cox proportional hazards for the 12 incident diseases + all-cause death

Model: transformer + BiLSTM over the patch embeddings with attention pooling (packed
sequences and masked pooling, so predictions do not depend on batch padding). The model
evaluated on test is a copy of the one from the best validation epoch:
AHI = lowest validation BCE (config `ahi_select_metric: auroc` for few-shot runs);
diagnosis = highest mean validation C-index over death and the diagnoses with events.
Selection criteria (config `selection`): 'all' / 'clean' = all / subject-disjoint
validation recordings. Few-shot options as in sleep_age.py.
Outputs (results_path): <run>_<task>_history.json, <run>_<task>_select-<criterion>_test.npz.

Usage:  python -m sleepbench.downstream.heads --task ahi --config <yaml> [--smoke]
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
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from torch.utils.data import DataLoader, Dataset

TASKS = {  # per-task training settings
    'ahi':       {'batch_div': 2, 'lr_div': 4, 'epochs': 40, 'patience': 8},
    'diagnosis': {'batch_div': 1, 'lr_div': 3, 'epochs': 40, 'patience': 5},
}
# Validation/test batch size. Outputs do not depend on it (packed LSTM, masked
# pooling); full-precision batches of 9 full nights ran out of memory on 48 GB.
EVAL_BATCH = 4


def study_id(path):
    return os.path.basename(path).split('.')[0]


def few_shot_ids(fs):
    """Study IDs of one few-shot subset (file names in the splits JSON)."""
    with open(fs['splits_path']) as f:
        return {study_id(n) for n in json.load(f)[str(fs['size'])][int(fs['rep'])]}


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

class HeadDataset(Dataset):
    """One item per recording: embeddings + recording-level labels."""

    def __init__(self, config, task, split, pretrain_type, keep_ids=None, repeat_to=None,
                 max_files=None):
        demo = pd.read_csv(config['demographics_labels_path'])
        d_ev = pd.read_csv(os.path.join(config['diagnosis_labels_path'], 'is_event.csv'))
        d_t = pd.read_csv(os.path.join(config['diagnosis_labels_path'], 'time_to_event.csv'))
        x_ev = pd.read_csv(os.path.join(config['death_labels_path'], 'is_event.csv'),
                           usecols=['Study ID', 'death'])
        x_t = pd.read_csv(os.path.join(config['death_labels_path'], 'time_to_event.csv'),
                          usecols=['Study ID', 'death'])
        ahi = pd.read_csv(config['ahi_labels_path'])
        self.diag_names = list(d_ev.columns[1:])
        assert list(d_t.columns[1:]) == self.diag_names

        def by_id(df):
            df = df.copy()
            df['Study ID'] = df['Study ID'].astype(str)
            return df.drop_duplicates('Study ID').set_index('Study ID')

        d_ev, d_t, x_ev, x_t, demo, ahi = map(by_id, (d_ev, d_t, x_ev, x_t, demo, ahi))
        # eligible: demographics, diagnosis, death (+ AHI for the AHI task)
        ok = set(demo.index) & set(d_ev.index) & set(d_t.index) & set(x_ev.index) & set(x_t.index)
        if task == 'ahi':
            ok &= set(ahi.index)
        label_dir = os.path.join(config['labels_path'], config['dataset'])
        ok &= {study_id(f) for f in os.listdir(label_dir) if f.endswith('.csv')}
        ok &= {study_id(f) for f in os.listdir(config['raw_data_dir'])}
        with open(config['split_path']) as f:
            ok &= {study_id(p) for p in json.load(f)[split]}

        emb = []
        for d, _, files in os.walk(os.path.join(config['embedding_path'], pretrain_type)):
            emb += [os.path.join(d, f) for f in files if f.endswith('.hdf5')]
        self.files = sorted(f for f in emb if study_id(f) in ok)
        if config.get('max_files'):
            self.files = self.files[:config['max_files']]
        if max_files:
            self.files = self.files[:max_files]
        if keep_ids is not None:
            sub = [f for f in self.files if study_id(f) in keep_ids]
            reps = max(int(repeat_to or 1) // max(len(sub), 1), 1)
            self.files = sub * reps
            print(f'few-shot: {len(sub)} of {len(keep_ids)} subset recordings eligible, '
                  f'x{reps} = {len(self.files)} items per epoch', flush=True)
        ids = [study_id(f) for f in self.files]

        self.labels = {
            'ahi': (ahi.loc[ids, 'ahi'].to_numpy() >= 15).astype(np.float32) if task == 'ahi'
            else np.zeros(len(ids), np.float32),
            'diag_event': d_ev.loc[ids, self.diag_names].to_numpy(np.float32),
            'diag_time': d_t.loc[ids, self.diag_names].to_numpy(np.float32),
            'death_event': x_ev.loc[ids, 'death'].to_numpy(np.float32),
            'death_time': x_t.loc[ids, 'death'].to_numpy(np.float32),
        }
        print(f'{split}: {len(self.files)} recordings', flush=True)

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        with h5py.File(self.files[i], 'r') as hf:
            x = np.stack([hf[k][:] for k in hf.keys()])
        x = torch.tensor(x, dtype=torch.float32)
        if x.ndim == 3:
            x = x.mean(dim=0)
        return x, {k: v[i] for k, v in self.labels.items()}, study_id(self.files[i])


def make_collate(max_len):
    def collate(batch):
        xs, ys, sids = zip(*batch)
        T = min(max(len(x) for x in xs), max_len)
        x = torch.zeros(len(xs), T, xs[0].shape[-1])
        pad = torch.ones(len(xs), T, dtype=torch.bool)
        for i, xi in enumerate(xs):
            s = min(len(xi), T)
            x[i, :s], pad[i, :s] = xi[:s], False
        y = {k: torch.tensor(np.stack([d[k] for d in ys])) for k in ys[0]}
        return x, pad, y, list(sids)
    return collate


# ---------------------------------------------------------------------------
# Model (SleepEventLSTMClassifier backbone, padding-invariant)
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


class HeadModel(nn.Module):
    def __init__(self, embed_dim, num_heads, num_layers, n_diag, pooling_head=4,
                 dropout=0.1, max_seq_length=8196):
        super().__init__()
        self.pos_enc = PositionalEncoding(max_seq_length, embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        enc = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout,
                                         batch_first=True, norm_first=True)
        self.transformer = nn.TransformerEncoder(enc, num_layers=num_layers)
        self.lstm = nn.LSTM(embed_dim, embed_dim // 2, num_layers=num_layers, batch_first=True,
                            dropout=dropout if num_layers > 1 else 0.0, bidirectional=True)
        self.pool = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=pooling_head,
                                               dropout=dropout, batch_first=True)
        self.fc_ahi = nn.Linear(embed_dim, 1)
        self.fc_death = nn.Linear(embed_dim, 1)
        self.fc_diagnosis = nn.Linear(embed_dim, n_diag)

    def forward(self, x, pad):
        T = x.size(1)
        x = self.transformer(self.norm(self.pos_enc(x)), src_key_padding_mask=pad)
        packed = pack_padded_sequence(x, (~pad).sum(1).cpu(), batch_first=True, enforce_sorted=False)
        x, _ = pad_packed_sequence(self.lstm(packed)[0], batch_first=True, total_length=T)
        h = self.pool(x, src_key_padding_mask=pad)
        keep = (~pad).unsqueeze(-1).to(h.dtype)
        h = (h * keep).sum(1) / keep.sum(1).clamp(min=1)
        return {'ahi': self.fc_ahi(h).squeeze(-1), 'death': self.fc_death(h),
                'diagnosis': self.fc_diagnosis(h)}


def cox_ph_loss(hazards, times, events):
    """Cox partial likelihood (negative log), (N, K) inputs."""
    times, idx = times.sort(dim=0, descending=True)
    hazards = hazards.gather(0, idx).float()
    events = events.gather(0, idx)
    log_cum = torch.logcumsumexp(hazards, dim=0)
    losses = -(hazards - log_cum) * events
    return (losses.sum(0) / (events.sum(0) + 1e-9)).mean()


def task_loss(task, out, y):
    if task == 'ahi':
        return F.binary_cross_entropy_with_logits(out['ahi'].float(), y['ahi'])
    return (cox_ph_loss(out['diagnosis'], y['diag_time'], y['diag_event'])
            + cox_ph_loss(out['death'], y['death_time'].unsqueeze(1), y['death_event'].unsqueeze(1)))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict(model, loader, device):
    """Per-study outputs and labels as numpy arrays (sorted by study ID)."""
    model.eval()
    rows = {}
    for x, pad, y, sids in loader:
        out = model(x.to(device), pad.to(device))
        out = {k: v.float().cpu().numpy() for k, v in out.items()}
        for i, s in enumerate(sids):
            rows[s] = ({k: v[i] for k, v in out.items()}, {k: v[i].numpy() for k, v in y.items()})
    sids = sorted(rows)
    out = {k: np.stack([rows[s][0][k] for s in sids]) for k in rows[sids[0]][0]}
    lab = {k: np.stack([rows[s][1][k] for s in sids]) for k in rows[sids[0]][1]}
    return sids, out, lab


def subset_metrics(task, sids, out, lab, keep):
    idx = np.array([i for i, s in enumerate(sids) if s in keep])
    o = {k: torch.tensor(v[idx]) for k, v in out.items()}
    y = {k: torch.tensor(v[idx]) for k, v in lab.items()}
    res = {'n': int(len(idx)), 'loss': float(task_loss(task, o, y))}
    if task == 'ahi':
        res['auroc'] = float(roc_auc_score(y['ahi'], o['ahi'])) if len(set(y['ahi'].tolist())) == 2 else float('nan')
    else:
        dt, de = lab['death_time'][idx], lab['death_event'][idx]
        res['death_cindex'] = float(concordance_index(dt, -out['death'][idx, 0], de)) if de.sum() else float('nan')
        cis = []
        for k in range(out['diagnosis'].shape[1]):
            e = lab['diag_event'][idx, k]
            if e.sum():
                cis.append(concordance_index(lab['diag_time'][idx, k], -out['diagnosis'][idx, k], e))
        res['diag_mean_cindex'] = float(np.mean(cis)) if cis else float('nan')
        res['mean_cindex'] = float(np.nanmean([res['death_cindex']] + cis))
    return res


def selection_score(task, metrics, ahi_metric='loss'):
    """Higher is better. AHI: BCE loss or AUROC
    (`ahi_select_metric: auroc`, used for few-shot runs, where the validation
    loss selected near-untrained epochs)."""
    if task == 'ahi':
        return metrics['auroc'] if ahi_metric == 'auroc' else -metrics['loss']
    return metrics['mean_cindex']


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=sorted(TASKS))
    ap.add_argument('--config', required=True)
    ap.add_argument('--smoke', action='store_true', help='60/40/40 recordings, 2 epochs')
    args = ap.parse_args()
    task, T = args.task, TASKS[args.task]
    config = yaml.safe_load(open(args.config))
    epochs = config.get('epochs_override', T['epochs'])
    patience = config.get('patience_override', T['patience'])
    run = f"{config['run_name']}_{task}"
    if args.smoke:
        config['max_files'], epochs, run = 60, 2, run + '_smoke'

    seed = config.get('seed', 42)
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mp = config['model_params']
    out_dir = config['results_path']
    os.makedirs(out_dir, exist_ok=True)
    bs = int(config.get('batch_override', config['batch_size'] // T['batch_div']))
    lr = config.get('lr_override', config['lr'] / T['lr_div'])
    print(f'run={run} task={task} pretrain_type={config["pretrain_type"]} batch={bs} lr={lr:.2e} '
          f'epochs={epochs} patience={patience} device={device}', flush=True)

    fs = config.get('few_shot')
    ds = {'pretrain': HeadDataset(config, task, 'pretrain', config['pretrain_type'],
                                  keep_ids=few_shot_ids(fs) if fs else None,
                                  repeat_to=fs.get('repeat_to', 1024) if fs else None),
          'validation': HeadDataset(config, task, 'validation', config['pretrain_type'],
                                    max_files=config.get('val_max_files')),
          'test': HeadDataset(config, task, 'test', config['pretrain_type'])}
    collate = make_collate(mp['max_seq_length'])
    nw = config['num_workers']
    train_dl = DataLoader(ds['pretrain'], batch_size=bs, shuffle=True, num_workers=nw,
                          collate_fn=collate, drop_last=True)
    val_dl = DataLoader(ds['validation'], batch_size=EVAL_BATCH, shuffle=False, num_workers=nw,
                        collate_fn=collate)
    test_dl = DataLoader(ds['test'], batch_size=EVAL_BATCH, shuffle=False, num_workers=nw,
                         collate_fn=collate)

    val_ids = [study_id(f) for f in ds['validation'].files]
    subsets = {'all': set(val_ids)}
    if config.get('val_clean_ids_path'):
        clean = set(open(config['val_clean_ids_path']).read().split()) & set(val_ids)
        subsets.update(clean=clean, overlap=set(val_ids) - clean)
        print(f'validation: {len(clean)} subject-disjoint / {len(val_ids)}', flush=True)
    crit = [c for c in config.get('selection', ['all', 'clean']) if c in subsets]
    stop_on = config.get('early_stop', crit[:1])

    model = HeadModel(mp['embed_dim'], mp['num_heads'], mp['num_layers'],
                      len(ds['pretrain'].diag_names), mp['pooling_head'], mp['dropout'],
                      mp['max_seq_length']).to(device)
    opt = optim.AdamW(model.parameters(), lr=lr)
    scaler = torch.cuda.amp.GradScaler(enabled=device.type == 'cuda')

    best = {c: {'score': -float('inf'), 'epoch': -1, 'state': None} for c in crit}
    history, patience_ctr = [], 0
    for epoch in range(epochs):
        t0 = time.time()
        model.train()
        tot, nb = 0.0, 0
        for x, pad, y, _ in train_dl:
            x, pad = x.to(device), pad.to(device)
            y = {k: v.to(device) for k, v in y.items()}
            opt.zero_grad()
            with torch.autocast('cuda', enabled=device.type == 'cuda'):
                loss = task_loss(task, model(x, pad), y)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(opt)
            scaler.update()
            tot += loss.item(); nb += 1

        del x, pad, y, loss
        opt.zero_grad(set_to_none=True)
        if device.type == 'cuda':
            torch.cuda.empty_cache()
        sids, out, lab = predict(model, val_dl, device)
        rec = {'epoch': epoch, 'train_loss': tot / max(nb, 1), 'secs': round(time.time() - t0),
               'gpu_peak_gb': round(torch.cuda.max_memory_allocated() / 2**30, 2)
               if device.type == 'cuda' else 0.0}
        rec.update({f'val_{k}': subset_metrics(task, sids, out, lab, v) for k, v in subsets.items()})
        history.append(rec)
        print(json.dumps(rec), flush=True)

        for c in crit:
            score = selection_score(task, rec[f'val_{c}'], config.get('ahi_select_metric', 'loss'))
            if score > best[c]['score']:
                best[c] = {'score': score, 'epoch': epoch,
                           'state': {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}}
        patience_ctr = 0 if any(best[c]['epoch'] == epoch for c in stop_on) else patience_ctr + 1
        if patience_ctr >= patience:
            print(f'early stopping at epoch {epoch}', flush=True)
            break

    json.dump({'config': config, 'task': task, 'history': history,
               'best_epoch': {c: best[c]['epoch'] for c in crit}},
              open(os.path.join(out_dir, f'{run}_history.json'), 'w'), indent=1)

    tested = {}
    for c in crit:
        if best[c]['epoch'] in tested:
            print(f'{c} selection picks the same epoch as {tested[best[c]["epoch"]]}; '
                  f'no extra test pass', flush=True)
            continue
        tested[best[c]['epoch']] = c
        model.load_state_dict(best[c]['state'])
        torch.save(best[c]['state'], os.path.join(out_dir, f'{run}_select-{c}_model.pt'))
        sids, out, lab = predict(model, test_dl, device)
        res = subset_metrics(task, sids, out, lab, set(sids))
        print(f'TEST select-{c} (epoch {best[c]["epoch"]}): {json.dumps(res)}', flush=True)
        np.savez_compressed(os.path.join(out_dir, f'{run}_select-{c}_test.npz'),
                            study_id=np.array(sids), diag_names=np.array(ds['test'].diag_names),
                            **{f'pred_{k}': v for k, v in out.items()}, **lab,
                            meta=json.dumps({'run': run, 'task': task, 'select': c,
                                             'epoch': best[c]['epoch'], 'test': res}))

        if args.smoke and c == crit[0]:
            dl1 = DataLoader(ds['test'], batch_size=1, shuffle=False, num_workers=nw, collate_fn=collate)
            _, out1, _ = predict(model, dl1, device)
            d = max(float(np.abs(out[k] - out1[k]).max()) for k in out)
            print(f'SMOKE batch-invariance: max|d output|={d:.2e}', flush=True)


if __name__ == '__main__':
    main()
