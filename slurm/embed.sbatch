#!/bin/bash
# Embed all recordings with one encoder (or a baseline).
# Usage: sbatch --export=METHOD=cl_pairwise slurm/embed.sbatch
#SBATCH --job-name=sb_embed
#SBATCH --partition=gpu
#SBATCH --gpus=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
python -m sleepbench.embed --method "$METHOD" --num_workers 10
