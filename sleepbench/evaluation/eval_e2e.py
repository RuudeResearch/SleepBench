"""Metrics of the end-to-end supervised baseline (same definitions and bootstrap as
eval_sleep_age / eval_heads).

    python -m sleepbench.evaluation.eval_e2e [--n_boot 1000]

Writes <output_root>/tables/e2e_metrics.csv and e2e_per_condition.csv.
"""
import argparse

import numpy as np
import pandas as pd
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

from .. import paths
from ..end_to_end.train_loop import TASK_RESULT_FILENAME
from .common import SENTINEL, STAGES, subject_map

CONDITIONS = ['Myocardial Infraction', 'Heart Failure', 'Atrial Fibrilation and Flutter',
              'General Atherosclerosis', 'Angina', 'Hypertension', 'Hypotension',
              'Pulmonary Heart Disease', 'Ischemic Heart Disease', 'Chronic Kidney Disease',
              'Type 2 Diabetes', 'Dementia']   # column names as spelled in the label files


def f1_cm(cm):
    tp = np.diagonal(cm, axis1=-2, axis2=-1).astype(float)
    return 2 * tp / np.maximum(2 * tp + (cm.sum(-2) - tp) + (cm.sum(-1) - tp), 1e-12)


def boots(sids, smap, n, seed=0):
    subs, idx = np.unique([smap.get(s, s) for s in sids], return_inverse=True)
    members = [np.flatnonzero(idx == i) for i in range(len(subs))]
    rng = np.random.default_rng(seed)
    return len(subs), [np.concatenate([members[i] for i in rng.integers(0, len(subs), len(subs))])
                       for _ in range(n)]


def ci(v):
    return np.percentile(v, 2.5), np.percentile(v, 97.5)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n_boot', type=int, default=1000)
    args = ap.parse_args()
    p = paths.load()
    load = lambda t: np.load(p.results / 'end_to_end' / t / TASK_RESULT_FILENAME[t], allow_pickle=True).item()
    smap = subject_map(p)
    rows, cond_rows = [], []

    r = load('sleep_age')
    sids = list(r['study_ids'])
    cms = np.zeros((len(sids), 5, 5))
    np.add.at(cms, (r['sleep_recording_ids'], np.array(r['sleep_targets']), np.array(r['sleep_predictions'])), 1)
    ae = np.abs(np.array(r['age_predictions']) - np.array(r['age_targets'])) * 100
    S, bs = boots(sids, smap, args.n_boot)
    f, fb = f1_cm(cms.sum(0)), np.stack([f1_cm(cms[b].sum(0)) for b in bs])
    rows.append(('macro_f1', f.mean(), *ci(fb.mean(1)), len(sids), S))
    rows += [(f'f1_{s}', f[i], *ci(fb[:, i]), len(sids), S) for i, s in enumerate(STAGES)]
    rows.append(('age_mae', ae.mean(), *ci([ae[b].mean() for b in bs]), len(sids), S))

    r = load('ahi')
    y, s = np.array(r['ahi_diagnosis_targets']), np.array(r['ahi_diagnosis_predictions'])
    S, bs = boots(list(r['study_ids']), smap, args.n_boot)
    rows.append(('auroc', roc_auc_score(y, s), *ci([roc_auc_score(y[b], s[b]) for b in bs]), len(y), S))
    rows.append(('ahi_prediction_sd', s.std(), np.nan, np.nan, len(y), S))

    r = load('death_diagnosis')
    d, g = r['death'], r['diagnosis']
    dt, de, dh = (np.asarray(d[k], float).ravel() for k in ('times', 'presence', 'hazards'))
    gt, gh = np.asarray(g['times'], float), np.asarray(g['hazards'], float)
    ge = np.where(gt > SENTINEL, np.asarray(g['presence'], float), 0.0)
    S, bs = boots(list(r['study_ids']), smap, args.n_boot)
    c = lambda t, h, e: concordance_index(t, -h, e)
    rows.append(('death_cindex', c(dt, dh, de), *ci([c(dt[b], dh[b], de[b]) for b in bs]), len(dt), S))
    point, boot = [], []
    for k, name in enumerate(CONDITIONS):
        v, bv = c(gt[:, k], gh[:, k], ge[:, k]), [c(gt[b, k], gh[b, k], ge[b, k]) for b in bs]
        cond_rows.append({'condition': name, 'cindex': v, 'lo': ci(bv)[0], 'hi': ci(bv)[1],
                          'n_events': int(ge[:, k].sum())})
        point.append(v)
        boot.append(bv)
    rows.append(('mean_diag_cindex', np.mean(point), *ci(np.mean(boot, axis=0)), len(dt), S))
    rows.append(('death_hazard_sd', dh.std(), np.nan, np.nan, len(dt), S))

    p.tables.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=['metric', 'value', 'lo', 'hi', 'n_recordings', 'n_individuals'])
    df.round(4).to_csv(p.tables / 'e2e_metrics.csv', index=False)
    pd.DataFrame(cond_rows).round(4).to_csv(p.tables / 'e2e_per_condition.csv', index=False)
    print(df.round(3).to_string(index=False))


if __name__ == '__main__':
    main()
