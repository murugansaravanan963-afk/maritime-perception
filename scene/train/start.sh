#!/bin/bash
# Unpack the scene datasets and submit the three trainings. Run on a Sol login node:
#   cd /scratch/smurug25/maritime-dataset && tar -xf scene_train.tar && bash scene/train/start.sh
set -euo pipefail
cd /scratch/smurug25/maritime-dataset
echo "unpacking datasets..."
tar -xf scene_lars_v1.tar
tar -xf uniri_seastate.tar
mkdir -p logs runs_scene
module load mamba/latest
source activate maritime
# GPU nodes may have no internet: fetch the pretrained weights here once (saved in ~/.cache/torch)
python -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1'); print('ResNet18 weights ready')"
python - <<'PY'
import json
s = json.load(open('scene/data/uniri_seastate/seastate_labels.json'))['records']
w = json.load(open('scene/data/scene_lars_v1/scene_labels.json'))
w = w['records'] if isinstance(w, dict) else w
count = lambda recs, key: {k: sum(r[key] == k for r in recs) for k in ('train', 'val', 'test')}
print('sea state images', count(s, 'split'))
print('water/horizon images', count(w, 'split'))
PY
for model in seastate water horizon; do
    sbatch -J "scene-$model" scene/train/train.sbatch "scene/train/train_$model.py" --out "runs_scene/$model"
done
squeue -u "$USER"
echo "Follow a job:  tail -f logs/scene-<model>-<job number>.out"
