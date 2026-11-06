"""LaTeX tables of the supplement (value line + CI line) from the result tables.

    python -m sleepbench.paper.tables

Writes <output_root>/tables/latex/: tableS3_sleep_stages.tex, tableS4_e2e.tex (if the
end-to-end metrics exist), tableS5_diagnosis.tex.
"""
import pandas as pd

from .. import paths
from ..evaluation.common import MODELS

P = paths.load()
TAB = P.tables
OUT = TAB / 'latex'
ORDER = ['cl_loo', 'cl_pairwise', 'dae', 'fft_dae', 'mae', 'mae_masked', 'fft_mae', 'fft_mae_masked',
         'base_time', 'base_fft']
LONG = {m[0]: m[1] for m in MODELS}
HEAD = {**{m[0]: m[2] for m in MODELS}, 'base_time': 'Baseline (T)', 'base_fft': 'Baseline (F)'}
CONDS = [('Myocardial Infraction', 'MI'), ('Heart Failure', 'HF'),
         ('Atrial Fibrilation and Flutter', 'AFF'), ('General Atherosclerosis', 'GATH'),
         ('Angina', 'ANG'), ('Hypertension', 'HTN'), ('Hypotension', 'HYPO'),
         ('Pulmonary Heart Disease', 'PHD'), ('Ischemic Heart Disease', 'IHD'),
         ('Chronic Kidney Disease', 'CKD'), ('Type 2 Diabetes', 'T2D'), ('Dementia', 'DEM')]
CI_NOTE = ('95\\% confidence intervals from 1,000 bootstrap resamples of individuals, '
           'retaining all recordings and epochs of a sampled individual.')


def cell(v, lo, hi, d=3):
    return f'{v:.{d}f}', f'\\scriptsize({lo:.{d}f}--{hi:.{d}f})'


def block(label, cells, last=False):
    vals = ' & '.join(c[0] for c in cells)
    cis = ' & '.join(c[1] for c in cells)
    end = ' \\\\' if last else ' \\\\[2pt]'
    return f'\\multirow{{2}}{{*}}{{{label}}}\n  & {vals} \\\\\n  & {cis}{end}\n'


def wrap(body, star=True):
    env = 'table*' if star else 'table'
    return f'\\begin{{{env}}}[h!]\n{body}\\end{{{env}}}\n'


def load():
    return tuple(pd.read_csv(TAB / f) for f in
                 ('sleep_age_metrics.csv', 'heads_metrics.csv', 'heads_per_condition.csv'))


def row(df, **kw):
    q = df
    for k, v in kw.items():
        q = q[q[k] == v]
    assert len(q) == 1, kw
    return q.iloc[0]


def tableS3(s):
    cols = [('f1_W', 'W'), ('f1_N1', 'N1'), ('f1_N2', 'N2'), ('f1_N3', 'N3'), ('f1_REM', 'REM'),
            ('macro_f1', 'Macro')]
    body = ('\\centering\n'
            '\\caption{Per-stage sleep staging F1 scores on the {\\dataset} test set (2,671 recordings '
            'from 1,969 individuals; 5-second resolution, unscored epochs excluded). Per-stage and macro '
            'F1 are computed from the confusion matrix pooled over all scored epochs of all test recordings. '
            + CI_NOTE + '}\n'
            '\\label{tab:sleep_stage_results}\n\\renewcommand{\\arraystretch}{0.85}\n'
            '\\resizebox{\\textwidth}{!}{\n\\begin{tabular}{l' + 'c' * len(cols) + '}\n\\toprule\n'
            'Model & ' + ' & '.join(c[1] for c in cols) + ' \\\\\n\\midrule\n')
    for i, m in enumerate(ORDER):
        r = row(s, selection='all', model=m)
        body += block(LONG[m], [cell(r[c], r[f'{c}_lo'], r[f'{c}_hi']) for c, _ in cols],
                      last=i == len(ORDER) - 1)
    body += '\\bottomrule\n\\end{tabular}\n}\n'
    return wrap(body)


def tableS5(h, pc):
    hh = lambda m: row(h, task='diagnosis', selection='all', model=m)
    body = ('\\centering\n'
            '\\caption{Per-condition C-index for all disease and mortality prediction tasks on the '
            '{\\dataset} test set (2,665 recordings from 1,965 individuals; prevalent cases censored at '
            'baseline). ' + CI_NOTE + ' Column headers: CL PW = contrastive learning pairwise; CL LOO = '
            'contrastive learning leave-one-out; MAE = masked autoencoder; DAE = denoising autoencoder; '
            'F = Fourier/frequency-domain objective; T = time-domain objective; M = masked patches; '
            'A = all patches. Diagnosis abbreviations: ANG = angina; AFF = atrial fibrillation and '
            'flutter; CKD = chronic kidney disease; DEM = dementia; GATH = general atherosclerosis; '
            'HF = heart failure; HTN = hypertension; HYPO = hypotension; IHD = ischemic heart disease; '
            'MI = myocardial infarction; PHD = pulmonary heart disease; T2D = type 2 diabetes.}\n'
            '\\label{tab:diagnosis_results}\n\\resizebox{\\textwidth}{!}{\n'
            '\\begin{tabular}{l' + 'c' * len(ORDER) + '}\n\\toprule\n'
            'Condition & ' + ' & '.join(HEAD[m] for m in ORDER) + ' \\\\\n\\midrule\n')
    body += block('Death', [cell(*hh(m)[['death_cindex', 'death_cindex_lo', 'death_cindex_hi']]) for m in ORDER])
    body += '\\midrule\n'
    for i, (key, lab) in enumerate(CONDS):
        body += block(lab, [cell(*row(pc, selection='all', model=m, condition=key)[['cindex', 'lo', 'hi']]) for m in ORDER],
                      last=i == len(CONDS) - 1)
    body += '\\midrule\n'
    body += block('Mean (12 diagnoses)', [cell(*hh(m)[['mean_diag_cindex', 'mean_diag_cindex_lo', 'mean_diag_cindex_hi']]) for m in ORDER], last=True)
    body += '\\bottomrule\n\\end{tabular}\n}\n'
    return wrap(body)


def tableS4():
    f = TAB / 'e2e_metrics.csv'
    if not f.exists():
        return None
    e = pd.read_csv(f).set_index('metric')
    pc = pd.read_csv(TAB / 'e2e_per_condition.csv').set_index('condition')
    rows = [('Sleep staging', 'Macro F1', 'macro_f1', 3), ('Age estimation', 'MAE (years)', 'age_mae', 2),
            ('AHI $\\geq$ 15 (binary)', 'AUROC', 'auroc', 3), ('Death', 'C-index', 'death_cindex', 3)]
    body = ('\\centering\n\\renewcommand{\\arraystretch}{0.85}\n'
            '\\caption{End-to-end supervised baseline without self-supervised pretraining. The encoder and '
            'downstream head were initialized randomly and trained jointly on each downstream task, using '
            'the same recordings as the frozen-encoder evaluation. ' + CI_NOTE + '}\n'
            '\\label{tab:e2e_baseline}\n\\begin{tabular}{llc}\n\\toprule\nTask & Metric & End-to-End \\\\\n\\midrule\n')
    for task, metric, key, d in rows:
        v, ci = cell(e.loc[key, 'value'], e.loc[key, 'lo'], e.loc[key, 'hi'], d)
        body += (f'\\multirow{{2}}{{*}}{{{task}}} & \\multirow{{2}}{{*}}{{{metric}}}\n  & {v} \\\\\n'
                 f'  & & {ci} \\\\[2pt]\n\\midrule\n')
    for key, lab in CONDS:
        v, ci = cell(pc.loc[key, 'cindex'], pc.loc[key, 'lo'], pc.loc[key, 'hi'])
        body += f'\\multirow{{2}}{{*}}{{{lab}}} & \\multirow{{2}}{{*}}{{C-index}}\n  & {v} \\\\\n  & & {ci} \\\\[2pt]\n'
    v, ci = cell(e.loc['mean_diag_cindex', 'value'], e.loc['mean_diag_cindex', 'lo'], e.loc['mean_diag_cindex', 'hi'])
    body += (f'\\midrule\n\\multirow{{2}}{{*}}{{Mean (12 diagnoses)}} & \\multirow{{2}}{{*}}{{C-index}}\n'
             f'  & {v} \\\\\n  & & {ci} \\\\\n\\bottomrule\n\\end{{tabular}}\n')
    return wrap(body)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    s, h, pc = load()
    out = {'tableS3_sleep_stages.tex': tableS3(s),
           'tableS5_diagnosis.tex': tableS5(h, pc), 'tableS4_e2e.tex': tableS4()}
    for name, tex in out.items():
        if tex:
            (OUT / name).write_text(tex)
            print('wrote', name)


if __name__ == '__main__':
    main()
