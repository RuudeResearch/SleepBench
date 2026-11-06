#!/bin/bash
# End-to-end baseline for one task (sleep_age, ahi or death_diagnosis); reads raw PSG.
# Usage: sbatch --export=TASK=sleep_age slurm/end_to_end.sbatch
#SBATCH --job-name=sb_e2e
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --constraint="GPU_MEM:40GB|GPU_MEM:48GB|GPU_MEM:80GB"   # peak use ~32 GB
#SBATCH --cpus-per-task=8
#SBATCH --mem=100G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
python -m sleepbench.end_to_end.train --task "$TASK"
