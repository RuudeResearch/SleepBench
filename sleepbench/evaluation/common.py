"""Shared pieces of the evaluation: model registry, subject IDs, multiple-testing correction."""
import glob
import json
import os

import numpy as np
import pandas as pd

# key, name, short name, colour, line style, marker. The first ten are the main models.
MODELS = [
    ('cl_loo', 'CL LOO', 'CL LOO', '#ece133', '--', 'o'),
    ('cl_pairwise', 'CL Pairwise', 'CL PW', '#ca9161', '-', 'o'),
    ('dae', 'DAE (Time)', 'DAE (T)', '#fbafe4', '--', 's'),
    ('fft_dae', 'DAE (Freq)', 'DAE (F)', '#cc78bc', '-', 's'),
    ('base_time', 'Baseline (Time)', 'Base (T)', '#d55e00', '-', 'x'),
    ('base_fft', 'Baseline (Freq)', 'Base (F)', '#de8f05', '--', 'x'),
    ('mae', 'MAE (Time, all patches)', 'MAE (T, A)', '#0173b2', '-', '^'),
    ('mae_masked', 'MAE (Time, masked patches)', 'MAE (T, M)', '#56b4e9', '--', '^'),
    ('fft_mae', 'MAE (Freq, all patches)', 'MAE (F, A)', '#029e73', '-.', 'D'),
    ('fft_mae_masked', 'MAE (Freq, masked patches)', 'MAE (F, M)', '#949494', ':', 'D'),
]
MAIN = [m[0] for m in MODELS]
# masking-rate ablation: 34% model -> its 75% counterpart (a main model)
MASK_PAIRS = {'mae_mask34': 'mae', 'mae_masked_mask34': 'mae_masked',
              'fft_mae_mask34': 'fft_mae', 'fft_mae_masked_mask34': 'fft_mae_masked'}
STAGES = ['W', 'N1', 'N2', 'N3', 'REM']
SENTINEL = float(np.float32(0.001))   # time-to-event <= this: prevalent at the PSG (censored at 0)


def subject_map(p):
    """Study ID -> individual (StanfordSleepBenchID). Without the file every recording is
    its own cluster, i.e. the bootstrap resamples recordings."""
    if not p.subject_ids_csv.exists():
        print(f'WARNING: {p.subject_ids_csv} not found; bootstrap resamples recordings, not individuals')
        return {}
    s = pd.read_csv(p.subject_ids_csv, dtype=str)
    return dict(zip(s['Study ID'], s['StanfordSleepBenchID']))


def bh(p):
    """Benjamini-Hochberg adjusted p-values."""
    p = np.asarray(p, float)
    n = len(p)
    order = np.argsort(p)
    q = np.minimum.accumulate((p[order] * n / np.arange(1, n + 1))[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(q, 0, 1)
    return out


def test_outputs(results_dir, run):
    """{selection criterion: (best epoch, npz)} for one run. A criterion that picked the same
    epoch as another has no test file of its own and shares that file."""
    best = json.load(open(os.path.join(results_dir, f'{run}_history.json')))['best_epoch']
    by_epoch = {int(json.loads(str(np.load(f)['meta']))['epoch']): f
                for f in glob.glob(os.path.join(results_dir, f'{run}_select-*_test.npz'))}
    return {crit: (ep, by_epoch[ep]) for crit, ep in best.items()}


def paired_p(d_obs, d_boot):
    """Two-sided paired-bootstrap p-value of an observed difference."""
    d_boot = np.asarray(d_boot)
    return float(np.mean(np.abs(d_boot - d_boot.mean()) >= abs(d_obs)))
