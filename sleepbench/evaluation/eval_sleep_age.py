"""Sleep staging and age metrics with 95% CIs from a bootstrap over individuals.

    python -m sleepbench.evaluation.eval_sleep_age [--group sleep_age] [--n_boot 1000]

Staging is evaluated at the 5-second resolution used for training; unscored patches are
excluded. Macro-F1 (and per-stage F1, accuracy, kappa) is computed from the confusion
matrix pooled over all scored patches of all test recordings. Age: MAE over recordings.
Each bootstrap resample draws individuals with replacement and keeps all their recordings
and patches; the same resamples are used for every model, so model differences are
tested pairwise (two-sided p, Benjamini-Hochberg over the 45 pairs of main models).
Masking rate: each 75% model vs its 34% counterpart (BH over the four variants).

Writes to <output_root>/tables/: <group>_metrics.csv, <group>_pairwise.csv,
<group>_masking.csv (one row per model x selection criterion).
"""
import argparse
import itertools
import json
import os

import numpy as np
import pandas as pd

from .. import paths
from .common import MAIN, MASK_PAIRS, STAGES, bh, paired_p, subject_map, test_outputs

C = len(STAGES)


def f1_from_cm(cm):
    """cm[..., true, pred] -> per-class F1 (..., C)."""
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    fp, fn = cm.sum(-2) - tp, cm.sum(-1) - tp
    with np.errstate(invalid='ignore', divide='ignore'):
        return np.where(2 * tp + fp + fn > 0, 2 * tp / (2 * tp + fp + fn), 0.0)


def metrics_from_cm(cm):
    f1 = f1_from_cm(cm)
    n = cm.sum((-2, -1)).astype(float)
    acc = np.diagonal(cm, axis1=-2, axis2=-1).sum(-1) / n
    pe = (cm.sum(-1) * cm.sum(-2)).sum(-1) / n ** 2
    out = {'macro_f1': f1.mean(-1), 'accuracy': acc, 'kappa': (acc - pe) / (1 - pe)}
    out.update({f'f1_{s}': f1[..., i] for i, s in enumerate(STAGES)})
    return out


def load_run(npz):
    """Per-recording confusion matrices and absolute age errors of one test output."""
    d = np.load(npz)
    sids, off, targets, probs = d['study_id'], d['offsets'], d['targets'], d['probs']
    cms = []
    for i in range(len(sids)):
        t, q = targets[off[i]:off[i + 1]], probs[off[i]:off[i + 1]]
        keep = t >= 0
        cms.append(np.bincount(t[keep].astype(np.int64) * C + q[keep].argmax(1),
                               minlength=C * C).reshape(C, C))
    return list(sids), np.stack(cms), np.abs(d['age_pred'] - d['age_true'])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--group', default='sleep_age')
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    p = paths.load()
    res = p.results / args.group
    smap = subject_map(p)
    runs = sorted(f.name[:-len('_history.json')] for f in res.glob('*_history.json')
                  if not f.name.endswith('_smoke_history.json'))

    by_sel, cache = {}, {}
    for run in runs:
        for sel, (ep, npz) in test_outputs(res, run).items():
            if npz not in cache:
                cache[npz] = load_run(npz)
            by_sel.setdefault(sel, []).append((run, ep, npz))

    rows, pair_rows, mask_rows = [], [], []
    for sel, grp in sorted(by_sel.items()):
        ref = cache[grp[0][2]][0]
        assert all(cache[npz][0] == ref for _, _, npz in grp), f'{sel}: test recordings differ'
        subjects, s_idx = np.unique([smap.get(s, s) for s in ref], return_inverse=True)
        S = len(subjects)
        rng = np.random.default_rng(args.seed)
        W = np.stack([np.bincount(rng.integers(0, S, S), minlength=S) for _ in range(args.n_boot)])
        Wr = W[:, s_idx]   # per-recording bootstrap weights, shared by all models

        stats = {}
        for run, ep, npz in grp:
            _, cms, ae = cache[npz]
            cm_subj = np.zeros((S, C, C))
            np.add.at(cm_subj, s_idx, cms)
            point = metrics_from_cm(cm_subj.sum(0))
            boot = metrics_from_cm(np.einsum('bs,sij->bij', W, cm_subj))
            point['age_mae'], boot['age_mae'] = ae.mean(), (Wr @ ae) / Wr.sum(1)
            stats[run] = (point, boot)
            row = {'selection': sel, 'model': run, 'best_val_epoch': ep, 'n_recordings': len(ref),
                   'n_individuals': S, 'n_scored_patches': int(cm_subj.sum())}
            for k in point:
                row[k], row[f'{k}_lo'], row[f'{k}_hi'] = point[k], *np.percentile(boot[k], [2.5, 97.5])
            rows.append(row)

        def diff(a, b, k):
            d_obs, d_boot = stats[a][0][k] - stats[b][0][k], stats[a][1][k] - stats[b][1][k]
            return d_obs, *np.percentile(d_boot, [2.5, 97.5]), paired_p(d_obs, d_boot)

        keys = list(stats[grp[0][0]][0])
        for a, b in itertools.combinations(sorted(m for m in stats if m in MAIN), 2):
            for k in keys:
                d, lo, hi, pv = diff(a, b, k)
                pair_rows.append({'selection': sel, 'model_a': a, 'model_b': b, 'metric': k,
                                  'delta': d, 'delta_lo': lo, 'delta_hi': hi, 'p': pv})
        for m34, m75 in MASK_PAIRS.items():
            if m34 in stats and m75 in stats:
                for k in keys:
                    d, lo, hi, pv = diff(m75, m34, k)
                    mask_rows.append({'selection': sel, 'model': m75, 'metric': k,
                                      'value_34': stats[m34][0][k], 'value_75': stats[m75][0][k],
                                      'delta_75_minus_34': d, 'delta_lo': lo, 'delta_hi': hi, 'p': pv})

    p.tables.mkdir(parents=True, exist_ok=True)
    met = pd.DataFrame(rows)
    met.round(4).to_csv(p.tables / f'{args.group}_metrics.csv', index=False)
    for name, t in (('pairwise', pair_rows), ('masking', mask_rows)):
        t = pd.DataFrame(t)
        if len(t):
            t['p_fdr'] = t.groupby(['selection', 'metric'])['p'].transform(bh)
            t.round(5).to_csv(p.tables / f'{args.group}_{name}.csv', index=False)
    bad = [(r['model'], k) for r in rows for k in ('macro_f1', 'f1_N3', 'age_mae')
           if not r[f'{k}_lo'] <= r[k] <= r[f'{k}_hi']]
    print(f'{len(met)} rows; point estimates outside their CI: {bad or "none"}')
    print(met[['selection', 'model', 'macro_f1', 'macro_f1_lo', 'macro_f1_hi', 'f1_N3', 'age_mae']]
          .round(3).to_string(index=False))


if __name__ == '__main__':
    main()
