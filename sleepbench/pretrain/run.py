"""Pre-train one encoder (multi-GPU via torchrun). From the repository root:

    torchrun --nproc_per_node=2 -m sleepbench.pretrain.run --method cl_pairwise

Methods are listed in configs.yaml. Checkpoints go to <output_root>/models/<method>/; the
best-validation checkpoints have `_val_loss_<x>.pt` in their name (see sleepbench.embed).
"""
import argparse
import importlib
import os
from pathlib import Path

import yaml

from .. import paths

CONFIGS = Path(__file__).with_name('configs.yaml')


def build_config(method, overrides=()):
    spec = yaml.safe_load(open(CONFIGS))
    if method not in spec['methods']:
        raise SystemExit(f"unknown method {method}; choose from {', '.join(spec['methods'])}")
    p = paths.load()
    if not p.split.exists():
        raise SystemExit(f'{p.split} not found: run `python -m sleepbench.setup` first.')
    cfg = {**spec['common'], **spec['methods'][method]}
    trainer = cfg.pop('trainer')
    cfg.update(split_path=str(p.split), channel_groups_path=str(p.channel_groups),
               save_path=str(p.models / method))
    for kv in overrides:
        k, v = kv.split('=', 1)
        cfg[k] = yaml.safe_load(v)
    return trainer, cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--method', required=True)
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE',
                    help='override config entries, e.g. --set num_workers=8 checkpoint_path=/x.pt')
    args = ap.parse_args()
    trainer, cfg = build_config(args.method, args.set)
    out = Path(cfg['save_path'])
    out.mkdir(parents=True, exist_ok=True)
    # one resolved config per process (avoids concurrent writes); rank 0's copy is the record
    cfg_file = out / f"config.rank{os.environ.get('RANK', '0')}.yaml"
    yaml.safe_dump(cfg, open(cfg_file, 'w'), sort_keys=False)
    importlib.import_module(f'.{trainer}', __package__).main_worker(None, None, str(cfg_file))


if __name__ == '__main__':
    main()
