"""Sleep stage label mapping from the StageName column of the label CSVs.

The StageNumber column is unreliable: it is -1 on every row of the 2012-2014
numeric-format SSC files and for R&K Stage 4 (SSC and SHHS). StageName holds
the scored hypnogram in all formats, so labels are derived from it here.

Classes: 0=W, 1=N1, 2=N2, 3=N3 (incl. R&K Stage 4), 4=REM, -1=unscored/ignored.
"""

import re

import numpy as np
import pandas as pd

STAGE_MAP = {
    # SSC named format
    'Wake': 0, 'Stage 1': 1, 'Stage 2': 2, 'Stage 3': 3, 'Stage 4': 3, 'REM': 4,
    'Unknown': -1,
    # SSC numeric format (2012-2014)
    '0': 0, '1': 1, '2': 2, '3': 3, '4': 3, '5': 4, '7': -1, '8': -1,
    # SHHS
    'Wake|0': 0, 'Stage 1 sleep|1': 1, 'Stage 2 sleep|2': 2,
    'Stage 3 sleep|3': 3, 'Stage 4 sleep|4': 3, 'REM sleep|5': 4,
    'Unscored|9': -1, 'Movement|6': -1,
}
_UNKNOWN_N = re.compile(r'^Unknown \(\d+\)$')


def stage_from_name(name):
    """Map one StageName string to a class index; raise on unseen strings."""
    if name in STAGE_MAP:
        return STAGE_MAP[name]
    if _UNKNOWN_N.match(name):
        return -1
    raise ValueError(f'Unmapped StageName: {name!r}')


def load_stage_labels(label_path):
    """Return the int64 class vector (one entry per 5-s row) for a label CSV."""
    names = pd.read_csv(label_path, usecols=['StageName'], dtype=str,
                        keep_default_na=False)['StageName']
    lut = {n: stage_from_name(n) for n in names.unique()}
    return names.map(lut).to_numpy(dtype=np.int64)


if __name__ == '__main__':
    expected = {
        'Wake': 0, '0': 0, 'Wake|0': 0,
        'Stage 1': 1, '1': 1, 'Stage 1 sleep|1': 1,
        'Stage 2': 2, '2': 2, 'Stage 2 sleep|2': 2,
        'Stage 3': 3, 'Stage 4': 3, '3': 3, '4': 3,
        'Stage 3 sleep|3': 3, 'Stage 4 sleep|4': 3,
        'REM': 4, '5': 4, 'REM sleep|5': 4,
        'Unknown': -1, 'Unknown (6)': -1, 'Unknown (8)': -1, 'Unknown (9)': -1,
        'Unknown (10)': -1, 'Unknown (11)': -1, 'Unknown (12)': -1,
        '7': -1, '8': -1, 'Unscored|9': -1, 'Movement|6': -1,
    }
    for k, v in expected.items():
        assert stage_from_name(k) == v, (k, stage_from_name(k), v)
    for bad in ['6', 'Stage 5', '']:
        try:
            stage_from_name(bad)
        except ValueError:
            continue
        raise AssertionError(f'{bad!r} should raise')
    print(f'OK: {len(expected)} label strings mapped as expected')
