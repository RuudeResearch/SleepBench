"""Write the run configs for all downstream experiments and one command list per group.

    python -m sleepbench.downstream.experiments

Writes <output_root>/runs/<group>/<run>.yaml and <output_root>/runs/<group>.txt (one shell
command per line; slurm/run_list.sbatch runs line $SLURM_ARRAY_TASK_ID). Groups:
  sleep_age        staging + age, 10 models + 4 masking-rate (34%) models (14 runs)
  heads            apnea and disease/mortality heads, same 14 models       (28 runs)
  few_shot_sleep   staging + age, 10 models x 6 sizes x 3 replicates      (180 runs)
  few_shot_ahi     apnea, 10 models x 6 sizes x 3 replicates              (180 runs)
  objective        joint vs staging-only vs age-only training (1 model)    (4 runs)
"""
import yaml

from .. import paths

MODELS = ['cl_pairwise', 'cl_loo', 'mae', 'mae_masked', 'fft_mae', 'fft_mae_masked',
          'dae', 'fft_dae', 'base_time', 'base_fft']
MASK34 = ['mae_mask34', 'mae_masked_mask34', 'fft_mae_mask34', 'fft_mae_masked_mask34']
FEW_SHOT_SIZES = (1, 8, 64, 256, 512, 1024)

HEAD = {'model_params': {'embed_dim': 512, 'num_heads': 4, 'num_layers': 1, 'num_classes': 5,
                         'pooling_head': 4, 'dropout': 0.3, 'max_seq_length': 8196},
        'seed': 42, 'batch_size': 18, 'lr': 0.0001, 'num_workers': 8, 'max_files': None}
SLEEP = {'patience': 15, 'epochs': 70}   # the heads script sets its own per-task values


def main():
    p = paths.load()
    data = {'dataset': 'ssc_stanford', 'raw_data_dir': str(p.ssc_hdf5_dir), 'split_path': str(p.split),
            'labels_path': str(p.sleep_stages), 'demographics_labels_path': str(p.demographics),
            'embedding_path': str(p.embeddings)}
    labels = {'ahi_labels_path': str(p.ahi), 'diagnosis_labels_path': str(p.diagnosis),
              'death_labels_path': str(p.death)}
    clean = {'val_clean_ids_path': str(p.val_subject_disjoint)} if p.val_subject_disjoint.exists() else {}
    sel = ['all', 'clean'] if clean else ['all']
    sel_sleep = ['all', 'age', 'clean', 'age-clean'] if clean else ['all', 'age']
    groups = {}

    def add(group, name, cfg, *cmds):
        d = p.runs / group
        d.mkdir(parents=True, exist_ok=True)
        cfg = {**cfg, 'run_name': name, 'results_path': str(p.results / group)}
        yaml.safe_dump(cfg, open(d / f'{name}.yaml', 'w'), sort_keys=False)
        groups.setdefault(group, []).extend(c.format(cfg=d / f'{name}.yaml') for c in cmds)

    sleep_cmd = 'python -m sleepbench.downstream.sleep_age --config {cfg}'
    for m in MODELS + MASK34:
        base = {**HEAD, **data, **clean, 'pretrain_type': m}
        add('sleep_age', m, {**base, **SLEEP, 'age_loss_weight': 10, 'selection': sel_sleep,
                             'early_stop': ['all', 'age']}, sleep_cmd)
        add('heads', m, {**base, **labels, 'selection': sel, 'early_stop': ['all']},
            *(f'python -m sleepbench.downstream.heads --task {t} --config {{cfg}}' for t in ('ahi', 'diagnosis')))

    # Few-shot: fixed subsets, 3 replicates; batch 9, lr 5e-5, 12 epochs, patience 6
    # (2 for 1 and 8 recordings), 1,000 validation recordings, subsets repeated to 1,024 per epoch.
    for m in MODELS:
        base = {**HEAD, **data, 'pretrain_type': m, 'val_max_files': 1000}
        for size in FEW_SHOT_SIZES:
            pat = 2 if size in (1, 8) else 6
            for rep in range(3):
                fs = {'size': size, 'rep': rep, 'repeat_to': 1024}
                add('few_shot_sleep', f'{m}_fs{size}_r{rep}',
                    {**base, 'age_loss_weight': 10, 'selection': ['all', 'age'], 'early_stop': ['all', 'age'],
                     'batch_size': 9, 'lr': 5e-5, 'epochs': 12, 'patience': pat,
                     'few_shot': {**fs, 'splits_path': str(paths.FEW_SHOT_SPLITS)}}, sleep_cmd)
                add('few_shot_ahi', f'{m}_fs{size}_r{rep}',
                    {**base, **labels, 'selection': ['all'], 'early_stop': ['all'],
                     'ahi_select_metric': 'auroc', 'batch_override': 9, 'lr_override': 5e-5,
                     'epochs_override': 12, 'patience_override': pat,
                     'few_shot': {**fs, 'splits_path': str(paths.FEW_SHOT_SPLITS_APNEA)}},
                    'python -m sleepbench.downstream.heads --task ahi --config {cfg}')

    # Training objective (one model): joint vs staging-only vs age-only.
    base = {**HEAD, **SLEEP, **data, **clean, 'pretrain_type': 'mae_masked'}
    add('objective', 'joint', {**base, 'age_loss_weight': 10, 'selection': sel}, sleep_cmd)
    add('objective', 'staging_only', {**base, 'age_loss_weight': 0, 'selection': sel}, sleep_cmd)
    add('objective', 'joint_select_age', {**base, 'age_loss_weight': 10,
                                          'selection': ['age', 'all']}, sleep_cmd)
    add('objective', 'age_only', {**base, 'age_loss_weight': 10, 'sleep_loss_weight': 0,
                                  'selection': ['age']}, sleep_cmd)

    for g, cmds in groups.items():
        (p.runs / f'{g}.txt').write_text('\n'.join(cmds) + '\n')
        print(f'{g:>10}: {len(cmds)} runs -> {p.runs / g}.txt')


if __name__ == '__main__':
    main()
