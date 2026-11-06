"""Figures of the paper from the result tables (run the evaluation scripts first).

    python -m sleepbench.paper.figures

Writes to <output_root>/figures/ (PDF and PNG): main_results (Fig. 2), few_shot_combined
(Fig. 3), pairwise_heatmap (Fig. S1), age_bins (Fig. S2; also tables/age_bins.csv),
diagnosis_by_condition (Fig. S3), masking_comparison_2x2 (Fig. S5).
"""
import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from .. import paths
from ..evaluation.common import MASK_PAIRS, MODELS, subject_map, test_outputs

P = paths.load()
TAB, OUT, RES = P.tables, P.figures, P.results
KEYS = [m[0] for m in MODELS]
NAME = {m[0]: m[1] for m in MODELS}
SHORT = {m[0]: m[2] for m in MODELS}
COL = {m[0]: m[3] for m in MODELS}
LS = {m[0]: m[4] for m in MODELS}
MK = {m[0]: m[5] for m in MODELS}


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / f'{name}.pdf', format='pdf', bbox_inches='tight')
    fig.savefig(OUT / f'{name}.png', dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'saved {name}')


def legend_patches():
    return [mpatches.Patch(color=COL[k], label=NAME[k]) for k in KEYS]


# ---------------------------------------------------------------------------
# Data
# ---------------------------------------------------------------------------

def main_metrics():
    s = pd.read_csv(TAB / 'sleep_age_metrics.csv')
    h = pd.read_csv(TAB / 'heads_metrics.csv')
    ssc = s
    out = {}
    for k in KEYS:
        a = ssc[(ssc.model == k) & (ssc.selection == 'all')].iloc[0]
        g = ssc[(ssc.model == k) & (ssc.selection == 'age')].iloc[0]
        ahi = h[(h.task == 'ahi') & (h.selection == 'all') & (h.model == k)].iloc[0]
        dg = h[(h.task == 'diagnosis') & (h.selection == 'all') & (h.model == k)].iloc[0]
        out[k] = {'sleep': (a.macro_f1, a.macro_f1_lo, a.macro_f1_hi),
                  'age': (g.age_mae, g.age_mae_lo, g.age_mae_hi),
                  'ahi': (ahi.auroc, ahi.auroc_lo, ahi.auroc_hi),
                  'diag': (dg.mean_diag_cindex, dg.mean_diag_cindex_lo, dg.mean_diag_cindex_hi),
                  'death': (dg.death_cindex, dg.death_cindex_lo, dg.death_cindex_hi)}
    return out


# ---------------------------------------------------------------------------
# Fig 2: main results
# ---------------------------------------------------------------------------

PANELS = [
    ('sleep', 'Sleep Stage Classification\n(Macro F1 ↑)', False, (0.6, 0.76), '.3f'),
    ('ahi', 'Sleep Apnea Classification\n(AUROC ↑)', False, (0.65, 0.9), '.3f'),
    ('age', 'Age Estimation\n(mean absolute error, years ↓)', True, (0, 11), '.2f'),
    ('diag', 'Disease Prediction\n(Mean C-index ↑)', False, (0.6, 0.8), '.3f'),
]


def fig_main(m):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, (key, title, lower, ylim, fmt) in zip(axes.flat, PANELS):
        order = sorted(KEYS, key=lambda k: m[k][key][0], reverse=not lower)
        pad = (ylim[1] - ylim[0]) * 0.012
        for i, k in enumerate(order):
            v, lo, hi = m[k][key]
            ax.bar(i, v, 0.7, color=COL[k], alpha=0.88, edgecolor='white')
            ax.errorbar(i, v, yerr=[[v - lo], [hi - v]], fmt='none', ecolor='#333333',
                        capsize=3, elinewidth=1)
            ax.text(i, hi + pad, format(v, fmt), ha='center', va='bottom', fontsize=7)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_ylim(*ylim)
        ax.set_xticks([])
        ax.grid(axis='y', alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    fig.legend(handles=legend_patches(), loc='upper center', bbox_to_anchor=(0.5, 1.1),
               ncol=5, fontsize=11)
    plt.tight_layout()
    save(fig, 'main_results')


# ---------------------------------------------------------------------------
# Fig 3: few-shot
# ---------------------------------------------------------------------------

def fig_few_shot():
    s = pd.read_csv(TAB / 'few_shot_summary.csv')
    panels = [('macro_f1', 'Sleep staging (macro F1 ↑)', (0.15, 0.75)),
              ('auroc', 'Sleep apnea (AUROC ↑)', (0.40, 0.85)),
              ('age_mae', 'Age estimation (MAE, years ↓)', (5, 26))]
    sizes = [1, 8, 64, 256, 512, 1024]
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    for ax, (metric, title, ylim) in zip(axes, panels):
        for k in KEYS:
            d = s[(s.model == k) & (s.metric == metric)].sort_values('size')
            ax.plot(d['size'], d['mean'], color=COL[k], linestyle=LS[k], marker=MK[k],
                    linewidth=1.6, markersize=5, label=NAME[k])
            ax.fill_between(d['size'], d['ci_lo'], d['ci_hi'], color=COL[k], alpha=0.06)
        ax.set_ylabel(title, fontsize=10)
        ax.set_ylim(*ylim)
        ax.grid(alpha=0.25)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    axes[-1].set_xscale('log')
    axes[-1].set_xticks(sizes)
    axes[-1].set_xticklabels([str(n) for n in sizes])
    axes[-1].set_xlabel('Number of labeled training recordings', fontsize=11)
    h, l = axes[0].get_legend_handles_labels()
    fig.legend(h, l, loc='upper center', bbox_to_anchor=(0.5, 1.06), ncol=5, fontsize=10)
    plt.tight_layout()
    save(fig, 'few_shot_combined')


# ---------------------------------------------------------------------------
# Fig S1: pairwise heatmap
# ---------------------------------------------------------------------------

def pair_matrix(df, run_of):
    N = len(KEYS)
    idx = {run_of(k): i for i, k in enumerate(KEYS)}
    d, p, r = np.full((N, N), np.nan), np.full((N, N), np.nan), np.zeros((N, N), bool)
    for _, row in df.iterrows():
        if row.model_a not in idx or row.model_b not in idx:
            continue
        i, j, sgn = idx[row.model_a], idx[row.model_b], 1
        if i > j:
            i, j, sgn = j, i, -1
        d[i, j], p[i, j], r[i, j] = sgn * row.delta, row.p_fdr, row.p_fdr < 0.05
    return d, p, r


def fig_pairwise():
    sp = pd.read_csv(TAB / 'sleep_age_pairwise.csv')
    hp = pd.read_csv(TAB / 'heads_pairwise.csv')
    sleep_run = lambda k: k
    head_run = lambda k: k
    sel = lambda df, **kw: df[np.logical_and.reduce([df[c] == v for c, v in kw.items()])]
    tasks = [
        ('Sleep Stage Classification\n(Macro F1 ↑)',
         pair_matrix(sel(sp, selection='all', metric='macro_f1'), sleep_run), 'ΔF1'),
        ('Sleep Apnea Classification\n(AUROC ↑)',
         pair_matrix(sel(hp, task='ahi', selection='all', metric='auroc'), head_run), 'ΔAUROC'),
        ('Age Estimation\n(mean absolute error, years ↓)',
         pair_matrix(sel(sp, selection='age', metric='age_mae'), sleep_run), 'Δmean abs. error (yr)'),
        ('Disease Prediction\n(Mean C-index ↑)',
         pair_matrix(sel(hp, task='diagnosis', selection='all', metric='mean_diag_cindex'), head_run), 'ΔC-index'),
    ]
    N = len(KEYS)
    labels = [SHORT[k] for k in KEYS]
    tri = np.tril(np.ones((N, N), bool))
    stars = lambda p, r: '' if not r else ('***' if p < 0.001 else '**' if p < 0.01 else '*')
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    for ax, (title, (d, p, r), clab) in zip(axes.flat, tasks):
        fin = d[~tri & np.isfinite(d)]
        vmax = max(np.nanpercentile(np.abs(fin), 98), 1e-6)
        sns.heatmap(d, ax=ax, mask=tri, cmap='RdBu_r', center=0, vmin=-vmax, vmax=vmax,
                    linewidths=0.5, linecolor='white', cbar_kws={'label': clab, 'shrink': 0.75},
                    xticklabels=labels, yticklabels=labels)
        for i in range(N):
            for j in range(i + 1, N):
                if not r[i, j]:
                    ax.add_patch(plt.Rectangle((j, i), 1, 1, color='#cccccc', alpha=0.65, zorder=2))
                s = stars(p[i, j], r[i, j])
                ax.text(j + 0.5, i + 0.5, f'{d[i, j]:+.2f}' + (f'\n{s}' if s else ''),
                        ha='center', va='center', fontsize=7.5, zorder=3,
                        color='white' if r[i, j] and abs(d[i, j]) > 0.6 * vmax else 'black',
                        fontweight='bold' if s else 'normal')
        ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
        ax.set_xlabel('Reference Model', fontsize=10)
        ax.set_ylabel('Model', fontsize=10)
        ax.tick_params(axis='x', rotation=45, labelsize=9)
        ax.tick_params(axis='y', rotation=0, labelsize=9)
    fig.legend(handles=[mpatches.Patch(facecolor='white', edgecolor='black', label='Significant (post-FDR BH)'),
                        mpatches.Patch(facecolor='#cccccc', edgecolor='black', label='Non-significant')],
               loc='lower center', ncol=2, fontsize=10, bbox_to_anchor=(0.5, 0.01),
               title='Cell significance (α=0.05, BH-FDR per task)', title_fontsize=10)
    plt.tight_layout(rect=[0, 0.04, 1, 1])
    save(fig, 'pairwise_heatmap')


# ---------------------------------------------------------------------------
# Fig S2: age bins; individuals resampled within each bin
# ---------------------------------------------------------------------------

EDGES = [0, 10, 20, 30, 40, 50, 60, 70, 80, 200]
BINS = ['0–9', '10–19', '20–29', '30–39', '40–49', '50–59', '60–69', '70–79', '≥80']


def compute_age_bins(n_boot=1000):
    smap = subject_map(P)
    rows = []
    for k in KEYS:
        d = np.load(test_outputs(RES / 'sleep_age', k)['age'][1])
        age, err = d['age_true'], np.abs(d['age_pred'] - d['age_true'])
        sids = np.array([smap.get(s, s) for s in d['study_id']])
        rng = np.random.default_rng(0)
        for lab, lo, hi in zip(BINS, EDGES[:-1], EDGES[1:]):
            m = (age >= lo) & (age < hi)
            subs, inv = np.unique(sids[m], return_inverse=True)
            e_s, n_s = np.bincount(inv, err[m]), np.bincount(inv)
            W = np.stack([np.bincount(rng.integers(0, len(subs), len(subs)), minlength=len(subs))
                          for _ in range(n_boot)])
            b = (W @ e_s) / (W @ n_s)
            rows.append({'model': k, 'bin': lab, 'n_recordings': int(m.sum()),
                         'n_individuals': len(subs), 'mae': err[m].mean(),
                         'lo': np.percentile(b, 2.5), 'hi': np.percentile(b, 97.5)})
        print(f'age bins: {k}', flush=True)
    df = pd.DataFrame(rows)
    df.round(4).to_csv(TAB / 'age_bins.csv', index=False)
    return df


def fig_age_bins(df):
    fig, ax = plt.subplots(figsize=(12, 6), dpi=150)
    x = np.arange(len(BINS))
    for k in KEYS:
        d = df[df.model == k].set_index('bin').loc[BINS]
        ax.plot(x, d.mae, label=NAME[k], color=COL[k], linestyle=LS[k], linewidth=2,
                marker='o', markersize=5, zorder=3)
        ax.fill_between(x, d.lo, d.hi, color=COL[k], alpha=0.12, zorder=2)
    ax.set_xticks(x)
    ax.set_xticklabels(BINS, rotation=30, fontsize=10)
    ax.set_xlabel('Age range (years)', fontsize=12)
    ax.set_ylabel('MAE (years)', fontsize=12)
    ax.set_title('Age Estimation MAE by Age Bin', fontsize=13, pad=10)
    ax.grid(True, linestyle='--', alpha=0.5, zorder=1)
    ax.legend(bbox_to_anchor=(0.5, 1.35), loc='upper center', fontsize=9, ncols=5, framealpha=0.9)
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(x)
    n = df[df.model == KEYS[0]].set_index('bin').loc[BINS].n_recordings
    ax2.set_xticklabels([f'n={v}' for v in n], fontsize=9, color='#555555')
    ax2.tick_params(axis='x', length=0)
    plt.tight_layout()
    save(fig, 'age_bins')


# ---------------------------------------------------------------------------
# Fig S3: C-index by condition
# ---------------------------------------------------------------------------

CONDS = [('Average', 'Avg'), ('death', 'Death'), ('Myocardial Infraction', 'MI'),
         ('Heart Failure', 'HF'), ('Atrial Fibrilation and Flutter', 'AFF'),
         ('General Atherosclerosis', 'GATH'), ('Angina', 'ANG'), ('Hypertension', 'HTN'),
         ('Hypotension', 'HYPO'), ('Pulmonary Heart Disease', 'PHD'),
         ('Ischemic Heart Disease', 'IHD'), ('Chronic Kidney Disease', 'CKD'),
         ('Type 2 Diabetes', 'T2D'), ('Dementia', 'DEM')]


def fig_diagnosis(m):
    pc = pd.read_csv(TAB / 'heads_per_condition.csv')
    pc = pc[pc.selection == 'all']
    fig, ax = plt.subplots(figsize=(18, 6), dpi=150)
    w, gap, xc = 0.055, 0.85, []
    for g, (cond, lab) in enumerate(CONDS):
        bars = []
        for k in KEYS:
            if cond == 'Average':
                v = m[k]['diag']
            elif cond == 'death':
                v = m[k]['death']
            else:
                r = pc[(pc.model == k) & (pc.condition == cond)].iloc[0]
                v = (r.cindex, r.lo, r.hi)
            bars.append((k, *v))
        bars.sort(key=lambda b: b[1], reverse=True)
        left = g * gap - len(bars) * w / 2
        for b, (k, v, lo, hi) in enumerate(bars):
            ax.bar(left + b * w, v, w, color=COL[k], alpha=0.88, linewidth=0)
            ax.errorbar(left + b * w, v, yerr=[[v - lo], [hi - v]], fmt='none', ecolor='#333333',
                        capsize=1.5, elinewidth=0.8, alpha=0.7)
        xc.append(g * gap)
    ax.set_xticks(xc)
    ax.set_xticklabels([c[1] for c in CONDS], rotation=40, ha='right', fontsize=9)
    ax.set_ylabel('C-index', fontsize=11)
    ax.set_title('C-index by Model and Condition (95% bootstrap CI)', fontsize=12, pad=8)
    ax.set_ylim(0.5, 0.92)
    ax.grid(axis='y', linestyle='--', alpha=0.5)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.legend(handles=legend_patches(), bbox_to_anchor=(0.5, 1.18), loc='upper center', ncols=5,
              fontsize=8.5, framealpha=0.9, handlelength=1.2, columnspacing=1.0)
    plt.tight_layout()
    save(fig, 'diagnosis_by_condition')


# ---------------------------------------------------------------------------
# Fig S5: masking 34 vs 75
# ---------------------------------------------------------------------------

def fig_masking():
    s = pd.read_csv(TAB / 'sleep_age_metrics.csv')
    h = pd.read_csv(TAB / 'heads_metrics.csv')
    pairs = list(MASK_PAIRS.items())

    def get(model, panel):
        if panel in ('sleep', 'age'):
            sel, col = ('all', 'macro_f1') if panel == 'sleep' else ('age', 'age_mae')
            r = s[(s.model == model) & (s.selection == sel)].iloc[0]
        else:
            task, col = ('ahi', 'auroc') if panel == 'ahi' else ('diagnosis', 'mean_diag_cindex')
            r = h[(h.task == task) & (h.selection == 'all') & (h.model == model)].iloc[0]
        return r[col], r[f'{col}_lo'], r[f'{col}_hi']

    panels = [('sleep', 'Sleep Stage Classification\n(Macro F1 ↑)', (0.6, 0.76), '.3f', 'Macro F1'),
              ('ahi', 'Sleep Apnea Classification\n(AUROC ↑)', (0.7, 0.9), '.3f', 'AUROC'),
              ('age', 'Age Estimation\n(MAE, years ↓)', (0, 11), '.2f', 'MAE (years)'),
              ('diag', 'Disease Prediction\n(Mean C-index ↑)', (0.6, 0.8), '.3f', 'C-index')]
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    wd = 0.35
    for ax, (panel, title, ylim, fmt, ylab) in zip(axes.flat, panels):
        pad = (ylim[1] - ylim[0]) * 0.015
        for xi, (m34, m75) in enumerate(pairs):
            for off, mod, kw in [(-wd / 2, m34, dict(alpha=0.55, hatch='//', edgecolor='white', linewidth=0.5)),
                                 (wd / 2, m75, dict(alpha=0.9))]:
                v, lo, hi = get(mod, panel)
                ax.bar(xi + off, v, wd, color=COL[m75], **kw)
                ax.errorbar(xi + off, v, yerr=[[v - lo], [hi - v]], fmt='none', color='#333333',
                            capsize=3, linewidth=0.8)
                ax.text(xi + off, hi + pad, format(v, fmt), ha='center', va='bottom', fontsize=6.5)
        ax.set_title(title, fontsize=11, fontweight='bold')
        ax.set_xticks([])
        ax.set_ylim(*ylim)
        ax.set_ylabel(ylab, fontsize=11)
        ax.grid(axis='y', alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
    handles = [mpatches.Patch(facecolor=COL[m75], label=NAME[m75], alpha=0.85) for _, m75 in pairs]
    handles += [mpatches.Patch(facecolor='#aaaaaa', alpha=0.55, hatch='//', edgecolor='white', label='34% masking'),
                mpatches.Patch(facecolor='#aaaaaa', alpha=0.9, label='75% masking')]
    fig.legend(handles=handles, loc='upper center', bbox_to_anchor=(0.5, 1.08), ncol=3, fontsize=11)
    plt.tight_layout()
    save(fig, 'masking_comparison_2x2')


def main():
    m = main_metrics()
    fig_main(m)
    fig_few_shot()
    fig_pairwise()
    fig_diagnosis(m)
    fig_masking()
    fig_age_bins(compute_age_bins())


if __name__ == '__main__':
    main()
