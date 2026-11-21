#!/bin/bash
# Current release: releases/maritime6_pilot_r1 (run preflight_release.py first).
# From the /scratch/smurug25/maritime-dataset directory:
# mkdir -p logs
# sbatch sol/train_fresh.sbatch <release_directory> <new_run_name> [n|s] [640|960|1280] [batch] [epochs]
#SBATCH -J maritime-fresh
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_eee59875463fall2026
#SBATCH -t 0-04:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -G a100:1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --export=NONE
set -euo pipefail
module load mamba/latest
source activate maritime
cd "${SLURM_SUBMIT_DIR}"
RELEASE=${1:?validated release directory required}
RUN_NAME=${2:?new run name required}
VARIANT=${3:-s}
IMAGE_SIZE=${4:-640}
BATCH=${5:-8}
EPOCHS=${6:-200}
case "$VARIANT" in n|s) ;; *) echo "Variant must be n or s" >&2; exit 2;; esac
python scripts/train_fresh.py --release "$RELEASE" --name "$RUN_NAME" \
  --weights "weights/yolov8${VARIANT}.pt" --imgsz "$IMAGE_SIZE" \
  --epochs "$EPOCHS" --batch "$BATCH" --workers 4 --device 0 --aug standard
