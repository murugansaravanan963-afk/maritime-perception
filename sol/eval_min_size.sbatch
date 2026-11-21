#!/bin/bash
# Score a trained model on VAL under minimum-object-size rules (detection range).
# From /scratch/smurug25/maritime-dataset:
#   sbatch sol/eval_min_size.sbatch releases/maritime6_pilot_r1 runs/detect/r1_y8s_1280/weights/best.pt 1280 r1_y8s_1280
#SBATCH -J maritime-minsize
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_eee59875463fall2026
#SBATCH -t 0-00:30:00
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
RELEASE=${1:?release directory required}
WEIGHTS=${2:?best.pt path required}
IMAGE_SIZE=${3:-1280}
NAME=${4:?short name for the report required}
cd scripts
python eval_min_size.py --release "../$RELEASE" --weights "../$WEIGHTS" --imgsz "$IMAGE_SIZE" --split val \
  --thresholds 0 4 6 8 10 12 16 --out "../reports/training/${NAME}_min_size_val.json"
