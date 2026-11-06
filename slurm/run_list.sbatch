#!/bin/bash
# Run one line of a command list written by `python -m sleepbench.downstream.experiments`.
# Usage: sbatch --array=1-14 --export=LIST=<output_root>/runs/sleep_age.txt slurm/run_list.sbatch
# (array size = number of lines; heads 28, few_shot_sleep 180, few_shot_ahi 180, objective 4)
#SBATCH --job-name=sb_downstream
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"   # peak use ~37 GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
cmd=$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$LIST")
echo "$cmd"
eval "$cmd"
