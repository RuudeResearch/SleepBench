"""Few-shot results: test metrics per run and mean with a Student-t 95% CI over the three
replicates of each subset size.

    python -m sleepbench.evaluation.eval_few_shot

Writes <output_root>/tables/few_shot_runs.csv and few_shot_summary.csv.
"""
import re

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import roc_auc_score

from .. import paths
from .common import test_outputs

NAME = re.compile(r'(.+)_fs(\d+)_r(\d)')


def macro_f1(targets, probs):
    keep = targets >= 0
    t, q = targets[keep].astype(int), probs[keep].argmax(1)
    cm = np.bincount(t * 5 + q, minlength=25).reshape(5, 5)
    tp = np.diag(cm).astype(float)
    f1 = 2 * tp / np.maximum(2 * tp + (cm.sum(0) - tp) + (cm.sum(1) - tp), 1)
    return f1.mean(), f1[3]


def main():
    p = paths.load()
    rows = []
    sd = p.results / 'few_shot_sleep'
    for h in sorted(sd.glob('*_history.json')):
        run = h.name[:-len('_history.json')]
        m, out = NAME.fullmatch(run), test_outputs(sd, run)
        d, a = np.load(out['all'][1]), np.load(out['age'][1])
        f1, f1_n3 = macro_f1(d['targets'], d['probs'])
        rows.append({'task': 'sleep_age', 'model': m[1], 'size': int(m[2]), 'rep': int(m[3]),
                     'macro_f1': f1, 'f1_N3': f1_n3, 'epoch_staging': out['all'][0],
                     'age_mae': float(np.abs(a['age_pred'] - a['age_true']).mean()),
                     'epoch_age': out['age'][0]})
    hd = p.results / 'few_shot_ahi'
    for h in sorted(hd.glob('*_ahi_history.json')):
        run = h.name[:-len('_history.json')]
        m, (ep, npz) = NAME.fullmatch(run[:-len('_ahi')]), test_outputs(hd, run)['all']
        d = np.load(npz)
        rows.append({'task': 'ahi', 'model': m[1], 'size': int(m[2]), 'rep': int(m[3]),
                     'auroc': roc_auc_score(d['ahi'], d['pred_ahi']), 'epoch_ahi': ep})
    runs = pd.DataFrame(rows)
    p.tables.mkdir(parents=True, exist_ok=True)
    runs.round(4).to_csv(p.tables / 'few_shot_runs.csv', index=False)

    summ = []
    for (task, model, size), g in runs.groupby(['task', 'model', 'size']):
        for metric in (['macro_f1', 'f1_N3', 'age_mae'] if task == 'sleep_age' else ['auroc']):
            v = g[metric].to_numpy()
            half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
            summ.append({'model': model, 'size': size, 'metric': metric, 'n_reps': len(v),
                         'mean': v.mean(), 'sd': v.std(ddof=1), 'ci_lo': v.mean() - half,
                         'ci_hi': v.mean() + half})
    summ = pd.DataFrame(summ)
    summ.round(4).to_csv(p.tables / 'few_shot_summary.csv', index=False)
    for metric in ['macro_f1', 'age_mae', 'auroc']:
        print(f'\n{metric} (mean over replicates)')
        print(summ[summ.metric == metric].pivot(index='model', columns='size', values='mean').round(3).to_string())


if __name__ == '__main__':
    main()
