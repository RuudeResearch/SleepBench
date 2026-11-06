"""All file locations, derived from one user file: <repo>/paths.yaml (or $SLEEPBENCH_PATHS).

paths.yaml needs two entries, `data_root` (the Stanford Sleep Bench download) and
`output_root` (anything this code writes); see paths.example.yaml.
"""
import os
from pathlib import Path
from types import SimpleNamespace

import yaml

REPO = Path(__file__).resolve().parents[1]
SPLITS = REPO / 'splits'
SSC_SPLIT = SPLITS / 'ssc_split.json'          # file names per split (fixed benchmark split)
FEW_SHOT_SPLITS = SPLITS / 'few_shot_splits.json'
FEW_SHOT_SPLITS_APNEA = SPLITS / 'few_shot_splits_w_apnea.json'
UNREADABLE = SPLITS / 'unreadable_recordings.txt'

LABEL_FILES = {
    'sleep_stages': 'sleep_stages',
    'demographics': 'demographics_info_10_14_24.csv',
    'ahi': 'ahi_stats_labels_not_normalized_outliers_removed.csv',
    'diagnosis': 'phewas_tte_prediction_10_18_24_num_labels_12',
    'death': 'phewas_tte_prediction_10_18_24_num_labels_1042',
}


def _first_dir(*candidates):
    for c in candidates:
        if Path(c).is_dir():
            return Path(c)
    return Path(candidates[0])


def load():
    cfg_file = Path(os.environ.get('SLEEPBENCH_PATHS', REPO / 'paths.yaml'))
    if not cfg_file.exists():
        raise SystemExit(f'{cfg_file} not found: copy paths.example.yaml to paths.yaml and edit it.')
    cfg = yaml.safe_load(open(cfg_file))
    data, out = Path(cfg['data_root']), Path(cfg['output_root'])
    # The download nests folders one level deeper than some local copies; accept both.
    ssc = Path(cfg['ssc_hdf5_dir']) if cfg.get('ssc_hdf5_dir') else \
        _first_dir(data / 'SSC_pre_2015' / 'SSC_pre_2015', data / 'SSC_pre_2015')
    labels = Path(cfg['labels_dir']) if cfg.get('labels_dir') else \
        _first_dir(data / 'labels' / 'labels', data / 'labels')
    subj = Path(cfg.get('subject_ids_csv') or data / 'SleepBench_StanfordSleepBenchID.csv')
    p = SimpleNamespace(
        data_root=data, ssc_hdf5_dir=ssc, labels_dir=labels, subject_ids_csv=subj,
        output_root=out,
        split=out / 'config' / 'ssc_split.json',               # written by `python -m sleepbench.setup`
        val_subject_disjoint=out / 'config' / 'validation_subject_disjoint_ids.txt',
        channel_groups=REPO / 'sleepbench' / 'pretrain' / 'channel_groups.json',
        models=out / 'models', embeddings=out / 'embeddings', runs=out / 'runs',
        results=out / 'results', tables=out / 'tables', figures=out / 'figures',
    )
    for k, v in LABEL_FILES.items():
        setattr(p, k, labels / v)
    return p
