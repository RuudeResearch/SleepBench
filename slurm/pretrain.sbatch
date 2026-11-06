#!/bin/bash
# Pre-train one encoder on 2 GPUs (A100/H100 80 GB used for the paper).
# Usage (from the repository root): sbatch --export=METHOD=cl_pairwise slurm/pretrain.sbatch
#SBATCH --job-name=sb_pretrain
#SBATCH --partition=gpu
#SBATCH --gpus=2
#SBATCH --cpus-per-task=20
#SBATCH --mem=200G
#SBATCH --time=2-00:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
torchrun --nproc_per_node=2 -m sleepbench.pretrain.run --method "$METHOD"
