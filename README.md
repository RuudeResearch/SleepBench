# Stanford Sleep Bench

Code for *Stanford Sleep Bench: Evaluating Polysomnography Pre-training Methods for Sleep
Foundation Models*. It runs the benchmark end to end from the data download:
self-supervised pre-training, embeddings, downstream evaluation, the end-to-end supervised
baseline and statistics.

## Setup

```bash
conda env create -f environment.yml && conda activate sleepbench
cp paths.example.yaml paths.yaml      # set data_root and output_root
python -m sleepbench.setup            # checks the download, writes the resolved split files
```

Data: request access at https://doi.org/10.60508/0ta5-v163 (Brain Data Science Platform).
`data_root` is the download folder. It contains the 16-channel HDF5 recordings and the
label files (sleep stages, demographics, AHI, incident diagnoses and death). If the download
includes `SleepBench_StanfordSleepBenchID.csv` (one ID per individual), the bootstrap
resamples individuals; without it, the bootstrap resamples recordings.

The benchmark split is fixed (`splits/ssc_split.json`): pre-training 12,955 recordings,
validation 1,500, test 3,015. The test set shares no individuals with the other splits.
All commands below are run from the repository root. Example Slurm scripts are in `slurm/`.
Adapt their partition and GPU constraints to your cluster.

## 1. Pre-training

```bash
torchrun --nproc_per_node=2 -m sleepbench.pretrain.run --method cl_pairwise
```

| method | model |
|---|---|
| `cl_pairwise`, `cl_loo` | CL Pairwise, CL LOO |
| `mae`, `mae_masked` | MAE (Time, all / masked patches), 75% masking |
| `fft_mae`, `fft_mae_masked` | MAE (Freq, all / masked patches), 75% masking |
| `dae`, `fft_dae` | DAE (Time), DAE (Freq) |
| `mae_mask34`, `mae_masked_mask34`, `fft_mae_mask34`, `fft_mae_masked_mask34` | MAE variants with 34% masking |

Settings are in `sleepbench/pretrain/configs.yaml`. Override any entry with
`--set key=value`, e.g. `--set num_workers=8`. MAE and DAE training stops early on the
loss of 100 randomly drawn validation recordings, and the checkpoint with the lowest
validation loss is used. For CL, the checkpoint saved after epoch 3 (0-based), batch 499 is
used. We used two A100 80 GB GPUs for 15–35 hours per method.
Checkpoints go to `<output_root>/models/<method>/`.

## 2. Embeddings

```bash
python -m sleepbench.embed --method cl_pairwise   # default checkpoint (or --checkpoint)
python -m sleepbench.embed --method base_fft      # baselines need no checkpoint
```

This step runs for the 12 pre-trained methods and the two baselines, `base_time` and
`base_fft` (no pre-training; fixed time and frequency features). Each model's embeddings
take about 230 GB.

## 3. Downstream evaluation (frozen embeddings)

```bash
python -m sleepbench.downstream.experiments       # writes run configs and command lists
sbatch --array=1-14 --export=LIST=<output_root>/runs/sleep_age.txt slurm/run_list.sbatch
```

| command list | runs | content |
|---|---|---|
| `sleep_age.txt` | 14 | sleep staging + age, jointly trained (10 models + 4 masking-rate models) |
| `heads.txt` | 28 | apnea (AHI ≥ 15) and disease/mortality (Cox) heads |
| `few_shot_sleep.txt`, `few_shot_ahi.txt` | 180 + 180 | 1–1,024 labelled recordings, 3 replicates |
| `objective.txt` | 4 | joint vs. staging-only vs. age-only training |

Each line is one command, e.g. `python -m sleepbench.downstream.heads --task ahi --config
<cfg>`, and can also be run directly. Runs use one GPU with at least 40 GB. On the GPUs
we used, few-shot runs took under an hour, heads 3–8 hours and sleep staging + age 5–25
hours.

The model evaluated on the test set comes from the epoch with the best validation
performance:
- sleep staging: macro-F1
- age: MAE
- apnea: binary cross-entropy (AUROC for few-shot runs)
- disease/mortality: mean C-index

Each output also contains the model selected on the subject-disjoint part of the validation
set (selection criteria `clean` / `age-clean`).

## 4. End-to-end supervised baseline

```bash
python -m sleepbench.end_to_end.train --task sleep_age   # also: ahi, death_diagnosis
```

The same encoder and head as above, randomly initialised and trained on raw PSG, using the
task-specific settings in `sleepbench/end_to_end/config.yaml`. Each task needs one GPU
(≥ 40 GB) for 2–7 hours.

## 5. Statistics

```bash
python -m sleepbench.evaluation.eval_sleep_age    # sleep staging and age
python -m sleepbench.evaluation.eval_heads        # apnea, disease and mortality
python -m sleepbench.evaluation.eval_e2e          # end-to-end baseline
python -m sleepbench.evaluation.eval_few_shot     # few-shot runs
```

Results are written to `<output_root>/tables/`.
- Sleep staging is evaluated per 5-second patch. Macro-F1 is computed from all scored
  patches of all test recordings pooled together; unscored patches are excluded.
- Diseases are predicted as incident outcomes: cases already present at the PSG are
  censored at time 0.
- 95% confidence intervals come from 1,000 bootstrap resamples of individuals, keeping all
  of their recordings and patches together.
- Models are compared pairwise on the same resamples, with Benjamini–Hochberg correction
  per task.

## Citation

If you use Stanford Sleep Bench, please cite:

> Kjaer MR, Thapa R, Ganjoo G, Moore H, Jennum PJ, Westover BM, Zou J, Mignot E, He B,
> Brink-Kjaer A. Stanford Sleep Bench: Evaluating Polysomnography Pre-training Methods for
> Sleep Foundation Models. arXiv:2512.09591 (2025). https://arxiv.org/abs/2512.09591

```bibtex
@article{kjaer2025sleepbench,
  title   = {Stanford Sleep Bench: Evaluating Polysomnography Pre-training Methods for Sleep Foundation Models},
  author  = {Kjaer, Magnus Ruud and Thapa, Rahul and Ganjoo, Gauri and Moore, Hyatt and Jennum, Poul Joergen and Westover, Brandon M. and Zou, James and Mignot, Emmanuel and He, Bryan and Brink-Kjaer, Andreas},
  journal = {arXiv preprint arXiv:2512.09591},
  year    = {2025},
  doi     = {10.48550/arXiv.2512.09591}
}
```

## Credits and license

Stanford Sleep Bench was developed at the Technical University of Denmark and Stanford
University (Mignot lab and Zou lab). The end-to-end baseline (`sleepbench/end_to_end/`) is
adapted from code by Rahul Thapa. MIT License (see `LICENSE`).
