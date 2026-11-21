#!/bin/bash
# Run one training script on a Sol A100. From /scratch/smurug25/maritime-dataset:
#   mkdir -p logs
#   sbatch scene/learn/train.sbatch scene/learn/train_seastate.py --out runs_scene/seastate
#SBATCH -J scene
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_eee59875463fall2026
#SBATCH -t 0-02:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -G a100:1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --export=NONE
set -euo pipefail
module load mamba/latest
source activate maritime
cd "${SLURM_SUBMIT_DIR}"
python "$@"
