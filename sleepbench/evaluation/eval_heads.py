"""Apnea and disease/mortality metrics with 95% CIs from a bootstrap over individuals.

    python -m sleepbench.evaluation.eval_heads [--n_boot 1000]

Apnea: AUROC and AUPRC of P(AHI >= 15). Mortality: Harrell's C-index of the predicted
hazard. Diseases: C-index per condition, with cases prevalent at the PSG (time-to-event
<= 0.001) censored at time 0; mean_diag_cindex = mean over the 12 diseases.
Bootstrap and paired tests as in eval_sleep_age (BH over the 45 main-model pairs;
masking: 75% vs 34%, BH over the four variants).

Writes to <output_root>/tables/: heads_metrics.csv, heads_per_condition.csv,
heads_pairwise.csv, heads_masking.csv.
"""
import argparse
import itertools

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import average_precision_score, roc_auc_score

from .. import paths
from .common import MAIN, MASK_PAIRS, SENTINEL, bh, paired_p, subject_map, test_outputs

MIN_EVENTS = 5   # conditions with fewer test events are skipped


def load(npz):
    d = np.load(npz)
    r = {'study_id': d['study_id'].astype(str)}
    if 'pred_ahi' in d.files and d['ahi'].std() > 0:
        r['ahi_score'], r['ahi'] = d['pred_ahi'].astype(float), d['ahi'].astype(float)
    r['death_h'] = d['pred_death'][:, 0].astype(float)
    r['death_t'], r['death_e'] = d['death_time'].astype(float), d['death_event'].astype(float)
    r['diag_h'] = d['pred_diagnosis'].astype(float)
    r['diag_t'] = d['diag_time'].astype(float)
    r['diag_e'] = np.where(r['diag_t'] > SENTINEL, d['diag_event'].astype(float), 0.0)
    r['diag_names'] = list(d['diag_names'])
    return r


def cidx(t, h, e):
    return concordance_index(t, -h, e)


def metrics(task, r, idx):
    if task == 'ahi':
        y, s = r['ahi'][idx], r['ahi_score'][idx]
        return {'auroc': roc_auc_score(y, s), 'auprc': average_precision_score(y, s)}
    out = {'death_cindex': cidx(r['death_t'][idx], r['death_h'][idx], r['death_e'][idx])}
    cs = []
    for k, name in enumerate(r['diag_names']):
        if r['diag_e'][:, k].sum() < MIN_EVENTS:   # judged on the full test set
            continue
        out[f'cindex_{name}'] = c = cidx(r['diag_t'][idx, k], r['diag_h'][idx, k], r['diag_e'][idx, k])
        cs.append(c)
    out['mean_diag_cindex'] = float(np.mean(cs))
    out['mean_cindex_with_death'] = float(np.mean(cs + [out['death_cindex']]))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_boot', type=int, default=1000)
    ap.add_argument('--seed', type=int, default=0)
    args = ap.parse_args()
    p = paths.load()
    res = p.results / 'heads'
    smap = subject_map(p)

    groups, cache = {}, {}
    for h in sorted(res.glob('*_history.json')):
        run = h.name[:-len('_history.json')]
        if run.endswith('_smoke'):
            continue
        model, task = run.rsplit('_', 1)
        for sel, (ep, npz) in test_outputs(res, run).items():
            groups.setdefault((task, sel), []).append((model, ep, npz))

    rows, cond_rows, pair_rows, mask_rows = [], [], [], []
    for (task, sel), grp in sorted(groups.items()):
        for _, _, npz in grp:
            if npz not in cache:
                cache[npz] = load(npz)
        ref = cache[grp[0][2]]['study_id']
        assert all((cache[npz]['study_id'] == ref).all() for _, _, npz in grp), f'{task}: test sets differ'
        subjects, s_idx = np.unique([smap.get(s, s) for s in ref], return_inverse=True)
        members = [np.flatnonzero(s_idx == i) for i in range(len(subjects))]
        rng = np.random.default_rng(args.seed)
        boots = [np.concatenate([members[i] for i in rng.integers(0, len(subjects), len(subjects))])
                 for _ in range(args.n_boot)]
        full = np.arange(len(ref))
        print(f'{task} {sel}: {len(grp)} models, {len(ref)} recordings, {len(subjects)} individuals', flush=True)

        stats = {}
        for model, ep, npz in grp:
            r = cache[npz]
            point = metrics(task, r, full)
            boot = pd.DataFrame([metrics(task, r, b) for b in boots])
            stats[model] = (point, boot)
            row = {'task': task, 'selection': sel, 'model': model, 'best_val_epoch': ep,
                   'n_recordings': len(ref), 'n_individuals': len(subjects)}
            for k, v in point.items():
                lo, hi = np.percentile(boot[k], [2.5, 97.5])
                if k.startswith('cindex_'):
                    name = k[len('cindex_'):]
                    cond_rows.append({'selection': sel, 'model': model, 'condition': name, 'cindex': v,
                                      'lo': lo, 'hi': hi,
                                      'n_events': int(r['diag_e'][:, r['diag_names'].index(name)].sum())})
                else:
                    row[k], row[f'{k}_lo'], row[f'{k}_hi'] = v, lo, hi
            rows.append(row)

        def diff(a, b, k):
            d_obs = stats[a][0][k] - stats[b][0][k]
            d_boot = (stats[a][1][k] - stats[b][1][k]).to_numpy()
            return d_obs, *np.percentile(d_boot, [2.5, 97.5]), paired_p(d_obs, d_boot)

        keys = [k for k in stats[grp[0][0]][0] if not k.startswith('cindex_')]
        for a, b in itertools.combinations(sorted(m for m in stats if m in MAIN), 2):
            for k in keys:
                d, lo, hi, pv = diff(a, b, k)
                pair_rows.append({'task': task, 'selection': sel, 'model_a': a, 'model_b': b, 'metric': k,
                                  'delta': d, 'delta_lo': lo, 'delta_hi': hi, 'p': pv})
        for m34, m75 in MASK_PAIRS.items():
            if m34 in stats and m75 in stats:
                for k in keys:
                    d, lo, hi, pv = diff(m75, m34, k)
                    mask_rows.append({'task': task, 'selection': sel, 'model': m75, 'metric': k,
                                      'value_34': stats[m34][0][k], 'value_75': stats[m75][0][k],
                                      'delta_75_minus_34': d, 'delta_lo': lo, 'delta_hi': hi, 'p': pv})

    p.tables.mkdir(parents=True, exist_ok=True)
    met = pd.DataFrame(rows)
    met.round(4).to_csv(p.tables / 'heads_metrics.csv', index=False)
    pd.DataFrame(cond_rows).round(4).to_csv(p.tables / 'heads_per_condition.csv', index=False)
    for name, t in (('pairwise', pair_rows), ('masking', mask_rows)):
        t = pd.DataFrame(t)
        if len(t):
            t['p_fdr'] = t.groupby(['task', 'selection', 'metric'])['p'].transform(bh)
            t.round(5).to_csv(p.tables / f'heads_{name}.csv', index=False)
    bad = [(r['model'], k) for r in rows for k in r if f'{k}_lo' in r and not r[f'{k}_lo'] <= r[k] <= r[f'{k}_hi']]
    print(f'point estimates outside their CI: {bad or "none"}')
    print(met[met.selection == 'all'].drop(columns=[c for c in met.columns if c.endswith(('_lo', '_hi'))])
          .round(3).to_string(index=False))


if __name__ == '__main__':
    main()
