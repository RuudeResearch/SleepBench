"""Cohort description per split (Table 2, Table S2) from the label files.

    python -m sleepbench.paper.dataset_tables

Demographics, apnea and outcome counts (diseases split into incident and prevalent at the
PSG), and the sleep stage distribution as a percentage of scored epochs, with unscored
recording time reported separately. Also N3 by age group.
Writes <output_root>/tables/dataset_by_split.csv and n3_by_age.csv.
"""
import json

import numpy as np
import pandas as pd

from .. import paths
from ..evaluation.common import SENTINEL, STAGES, subject_map
from ..labels import load_stage_labels

SPLITS = ['pretrain', 'validation', 'test']
CONDITIONS = {'Myocardial Infraction': 'Myocardial infarction', 'Heart Failure': 'Heart failure',
              'Atrial Fibrilation and Flutter': 'Atrial fibrillation and flutter',
              'General Atherosclerosis': 'General atherosclerosis', 'Angina': 'Angina',
              'Hypertension': 'Hypertension', 'Hypotension': 'Hypotension',
              'Pulmonary Heart Disease': 'Pulmonary heart disease',
              'Ischemic Heart Disease': 'Ischemic heart disease',
              'Chronic Kidney Disease': 'Chronic kidney disease', 'Type 2 Diabetes': 'Type 2 diabetes',
              'Dementia': 'Dementia'}   # label-file column -> display name
AGE_BINS, AGE_LABELS = [0, 13, 18, 40, 60, 200], ['<13', '13-17', '18-39', '40-59', '60+']


def stage_counts(p, ids):
    """Per-recording counts of scored stages, unscored epochs and leading unscored epochs."""
    rows = []
    for sid in ids:
        f = p.sleep_stages / 'ssc_stanford' / f'{sid}.csv'
        if not f.exists():
            continue
        y = load_stage_labels(f)
        scored = np.flatnonzero(y >= 0)
        rows.append({'study_id': sid, 'n_epochs': len(y), 'unscored': int((y < 0).sum()),
                     'leading_unscored': int(scored[0]) if len(scored) else len(y),
                     **{s: int((y == i).sum()) for i, s in enumerate(STAGES)}})
    return pd.DataFrame(rows).set_index('study_id')


def main():
    p = paths.load()
    split = {k: [f.split('.')[0] for f in v] for k, v in json.load(open(paths.SSC_SPLIT)).items()}
    rd = lambda f: pd.read_csv(f, dtype={'Study ID': str}).drop_duplicates('Study ID').set_index('Study ID')
    demo, ahi = rd(p.demographics), rd(p.ahi)
    dx_event, dx_time = rd(p.diagnosis / 'is_event.csv'), rd(p.diagnosis / 'time_to_event.csv')
    death = rd(p.death / 'is_event.csv')
    smap = subject_map(p)
    out, n3_age = {}, []
    for s in SPLITS:
        ids = split[s]
        d = demo.reindex(ids).dropna(how='all')
        age = d['Age at Study Date'].astype(float)
        bmi = d['Recent BMI'].astype(float)
        bmi = bmi[(bmi >= 10) & (bmi <= 80)]
        col = {'Recordings': len(ids),
               'Individuals': len({smap[i] for i in ids if i in smap}) if smap else np.nan,
               'Age, mean ± SD [range]': f'{age.mean():.1f} ± {age.std(ddof=0):.1f} [{age.min():.1f}-{age.max():.1f}]',
               'Age < 18 years, n (%)': f'{(age < 18).sum()} ({100 * (age < 18).mean():.1f}%)',
               'Male, %': f"{100 * d['Gender'].str.lower().isin(['m', 'male']).mean():.1f}",
               'BMI, mean ± SD': f'{bmi.mean():.1f} ± {bmi.std(ddof=0):.1f}'}
        for c in ('Race', 'Ethnicity'):
            for k, v in d[c].fillna('Unknown').value_counts().items():
                col[f'{c}: {k}, n (%)'] = f'{v} ({100 * v / len(d):.1f}%)'
        a = ahi.reindex(ids)['ahi'].dropna()
        col['AHI >= 15, n (% of recordings with AHI)'] = f'{(a >= 15).sum()} ({100 * (a >= 15).mean():.1f}%)'
        ev, tt = dx_event.reindex(ids), dx_time.reindex(ids)
        for c, name in CONDITIONS.items():
            pos = ev[c] == 1
            prev = pos & (tt[c] <= SENTINEL)
            col[f'{name}: incident'] = int((pos & ~prev).sum())
            col[f'{name}: prevalent'] = int(prev.sum())
        col['Deaths'] = int((death.reindex(ids)['death'] == 1).sum())
        sc = stage_counts(p, ids)
        n_scored = sc[STAGES].sum().sum()
        col['Recordings with sleep stage labels'] = len(sc)
        for st in STAGES:
            col[f'{st}, % of scored epochs'] = round(100 * sc[st].sum() / n_scored, 1)
        col['Unscored, % of recording time'] = round(100 * sc['unscored'].sum() / sc['n_epochs'].sum(), 1)
        col['Unscored epochs before scoring starts, %'] = round(100 * sc['leading_unscored'].sum() / sc['unscored'].sum(), 1)
        col['Leading unscored period, median minutes'] = sc['leading_unscored'].median() * 5 / 60
        col['Recordings entirely unscored'] = int((sc[STAGES].sum(1) == 0).sum())
        out[s] = col
        n3_age.append(sc.join(age.rename('age')))
    p.tables.mkdir(parents=True, exist_ok=True)
    t = pd.DataFrame(out)
    t.to_csv(p.tables / 'dataset_by_split.csv')
    print(t.to_string())

    sc = pd.concat(n3_age)
    sc['age_group'] = pd.cut(sc['age'], AGE_BINS, right=False, labels=AGE_LABELS)
    g = sc.groupby('age_group', observed=True)
    n3 = pd.DataFrame({'recordings': g.size(),
                       'N3, % of scored epochs': 100 * g['N3'].sum() / g[STAGES].sum().sum(1),
                       'recordings without N3, %': 100 * g['N3'].apply(lambda x: (x == 0).mean())})
    n3.round(1).to_csv(p.tables / 'n3_by_age.csv')
    print(n3.round(1).to_string())


if __name__ == '__main__':
    main()
