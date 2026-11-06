"""Train and evaluate the end-to-end supervised baseline for one task:

    python -m sleepbench.end_to_end.train --task sleep_age        # staging + age
    python -m sleepbench.end_to_end.train --task ahi              # AHI >= 15
    python -m sleepbench.end_to_end.train --task death_diagnosis  # 12 diseases + death

Adapted from R. Thapa's end-to-end baseline. Settings per task are in config.yaml; the
epoch with the lowest validation loss is evaluated on the test set. Output:
<output_root>/results/end_to_end/<task>/<task results>.npy (per-recording predictions).
"""
import argparse
from pathlib import Path

import yaml

from .. import paths
from .train_loop import (evaluate_and_save, save_checkpoint, setup_training,
                         task_checkpoint_path, task_output_path, train)

TASKS = ('sleep_age', 'ahi', 'death_diagnosis')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task', required=True, choices=TASKS)
    ap.add_argument('--set', nargs='*', default=[], metavar='KEY=VALUE', help='override config entries')
    args = ap.parse_args()
    p = paths.load()
    spec = yaml.safe_load(open(Path(__file__).with_name('config.yaml')))
    cfg = {**spec['common'], **spec['tasks'][args.task]}
    cfg.update(labels_path=str(p.sleep_stages), demographics_labels_path=str(p.demographics),
               ahi_labels_path=str(p.ahi), diagnosis_labels_path=str(p.diagnosis),
               death_labels_path=str(p.death), split_path=str(p.split),
               channel_groups_path=str(p.channel_groups), raw_data_dir=str(p.ssc_hdf5_dir),
               exclude_ids_path=str(paths.UNREADABLE), save_path=str(p.results / 'end_to_end'),
               run_name=args.task)
    for kv in args.set:
        k, v = kv.split('=', 1)
        cfg[k] = yaml.safe_load(v)
    cfg_file = p.runs / 'end_to_end' / f'{args.task}.yaml'
    cfg_file.parent.mkdir(parents=True, exist_ok=True)
    yaml.safe_dump(cfg, open(cfg_file, 'w'), sort_keys=False)

    config, device, model, train_loader, val_loader, test_loader = setup_training(str(cfg_file), task=args.task)
    model, best_val, optimizer, scaler = train(model, train_loader, val_loader, config, device,
                                               task=args.task, patience=config['patience'])
    print(f'Best validation loss: {best_val:.4f}')
    save_checkpoint(model, optimizer, scaler, config, path=task_checkpoint_path(config, args.task))
    evaluate_and_save(model, test_loader, output_path=task_output_path(config, args.task),
                      device=device, task=args.task, config=config)


if __name__ == '__main__':
    main()
