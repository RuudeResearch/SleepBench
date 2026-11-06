"""Check the data download and write the resolved split files. Run once:

    python -m sleepbench.setup

Writes <output_root>/config/ssc_split.json (absolute HDF5 paths per split) and, if the
subject ID file is present, the subject-disjoint validation subset used for the
model-selection sensitivity analysis.
"""
import json

import pandas as pd

from . import paths


def main():
    p = paths.load()
    missing = [k for k in ('ssc_hdf5_dir', *paths.LABEL_FILES) if not getattr(p, k).exists()]
    if missing:
        raise SystemExit('Not found: ' + ', '.join(f'{k} ({getattr(p, k)})' for k in missing))

    split = json.load(open(paths.SSC_SPLIT))
    present = {f.name for f in p.ssc_hdf5_dir.glob('*.hdf5')}
    for name, files in split.items():
        n_missing = sum(f not in present for f in files)
        print(f'{name:>10}: {len(files):6d} recordings' + (f', {n_missing} missing' if n_missing else ''))
    p.split.parent.mkdir(parents=True, exist_ok=True)
    json.dump({k: [str(p.ssc_hdf5_dir / f) for f in v] for k, v in split.items()},
              open(p.split, 'w'), indent=1)
    print(f'wrote {p.split}')

    if not p.subject_ids_csv.exists():
        print(f'{p.subject_ids_csv} not found: bootstrap will resample recordings, and the '
              'subject-disjoint validation subset is not created.')
        return
    subj = pd.read_csv(p.subject_ids_csv, dtype=str).set_index('Study ID')['StanfordSleepBenchID']
    sid = lambda files: [f.split('.')[0] for f in files]
    val, pre = sid(split['validation']), sid(split['pretrain'])
    pre_subjects = set(subj.reindex(pre).dropna())
    clean = sorted(s for s in val if s in subj.index and subj[s] not in pre_subjects)
    p.val_subject_disjoint.write_text('\n'.join(clean) + '\n')
    print(f'subject-disjoint validation subset: {len(clean)} of {len(val)} recordings '
          f'-> {p.val_subject_disjoint}')


if __name__ == '__main__':
    main()
