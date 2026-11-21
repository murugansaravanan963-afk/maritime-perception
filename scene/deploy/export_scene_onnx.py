"""Export the 3 scene models to ONNX for the Hailo compiler (and ONNX Runtime on the Pi).

Run on the laptop from D:/Projects/maritime-dataset:
  python scene/deploy/export_scene_onnx.py --water runs_scene/water/last.pt --horizon runs_scene/horizon/last.pt \
      --seastate runs_scene/seastate_v2/best.pt --out scene_export

Writes scene_export/: water.onnx, horizon.onnx, seastate.onnx, scene_meta.json, and calib/ (images the
Hailo compiler uses to pick INT8 ranges). Every ONNX file is checked against PyTorch on real images.
The water model is exported WITHOUT its last x8 upsampling: it outputs 1/8-size logits (3x48x80) and
the Pi upsamples them (scene_common.water_mask). Same result, 64x less data out of the Hailo.
"""
import argparse
import importlib.util
import json
import random
import shutil
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import scene_common as sc


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, HERE.parent / 'train' / f'{name}.py')
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class WaterNoUpsample(nn.Module):
    """WaterNet.forward without the final F.interpolate(scale_factor=8)."""
    def __init__(self, net):
        super().__init__()
        self.net = net

    def forward(self, x):
        n = self.net
        x = n.start(x)
        f2 = n.layer2(x)
        f3 = n.layer3(f2)
        f4 = n.layer4(f3)
        p = n.side4(f4)
        p = F.interpolate(p, scale_factor=2) + n.side3(f3)
        p = F.interpolate(p, scale_factor=2) + n.side2(f2)
        return n.head(p)


def load_weights(model, path):
    state = torch.load(path, map_location='cpu')
    model.load_state_dict(state)
    return model.eval()


def sample_images(lars_dir, uniri_dir, count):
    """A few real val images for the ONNX check: (letterboxed 640x384 RGB, 224 sea patches RGB)."""
    lars = json.load(open(lars_dir / 'scene_labels.json'))
    lars = lars['records'] if isinstance(lars, dict) else lars
    scene = [cv2.cvtColor(cv2.imread(str(lars_dir / r['image'])), cv2.COLOR_BGR2RGB)
             for r in [r for r in lars if r['split'] == 'val'][:count]]
    uniri = json.load(open(uniri_dir / 'seastate_labels.json'))['records']
    sea = []
    for r in [r for r in uniri if r['split'] == 'test'][::97][:count]:
        img = cv2.cvtColor(cv2.imread(str(uniri_dir / r['image'])), cv2.COLOR_BGR2RGB)
        c = (img.shape[0] - sc.SEA_SIZE) // 2
        sea.append(img[c:c + sc.SEA_SIZE, c:c + sc.SEA_SIZE])
    return scene, sea


def export(model, name, size, out_dir, output_name, check_images):
    h, w = size
    path = out_dir / f'{name}.onnx'
    torch.onnx.export(model, torch.zeros(1, 3, h, w), str(path), input_names=['images'], output_names=[output_name],
                      opset_version=13, dynamo=False, do_constant_folding=True)
    import onnxruntime as ort
    session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])
    worst = 0.0
    for rgb in check_images:
        x = sc.to_onnx_input(rgb)
        with torch.no_grad():
            ref = model(torch.from_numpy(x)).numpy()
        got = session.run(None, {'images': x})[0]
        worst = max(worst, float(np.abs(ref - got).max()))
    shape = list(session.get_outputs()[0].shape)
    print(f'{name:9s} -> {path}  output {shape}  max |torch - onnx| = {worst:.2e}', flush=True)
    if worst > 1e-3:
        raise SystemExit(f'{name}: ONNX output differs from PyTorch by {worst}')
    return {'onnx': path.name, 'input': [1, 3, h, w], 'output': shape, 'max_abs_diff_vs_torch': worst}


def write_calib(lars_dir, uniri_dir, out_dir, count):
    """Calibration images: letterboxed LaRS train frames (water, horizon), UNIRI train 224 crops (sea state)."""
    rng = random.Random(0)
    lars = json.load(open(lars_dir / 'scene_labels.json'))
    lars = lars['records'] if isinstance(lars, dict) else lars
    picks = [r for r in lars if r['split'] == 'train']
    rng.shuffle(picks)
    (out_dir / 'calib' / 'scene').mkdir(parents=True, exist_ok=True)
    for r in picks[:count]:
        shutil.copy(lars_dir / r['image'], out_dir / 'calib' / 'scene' / Path(r['image']).name)
    uniri = [r for r in json.load(open(uniri_dir / 'seastate_labels.json'))['records'] if r['split'] == 'train']
    rng.shuffle(uniri)
    (out_dir / 'calib' / 'seastate').mkdir(parents=True, exist_ok=True)
    for r in uniri[:count]:
        img = cv2.imread(str(uniri_dir / r['image']))
        c = (img.shape[0] - sc.SEA_SIZE) // 2
        cv2.imwrite(str(out_dir / 'calib' / 'seastate' / Path(r['image']).name),
                    img[c:c + sc.SEA_SIZE, c:c + sc.SEA_SIZE])
    print(f'calibration images: {min(count, len(picks))} scene, {min(count, len(uniri))} sea state', flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--water', type=Path, required=True)
    ap.add_argument('--horizon', type=Path, required=True)
    ap.add_argument('--seastate', type=Path, required=True)
    ap.add_argument('--lars', type=Path, default=Path('scene/data/scene_lars_v1'))
    ap.add_argument('--uniri', type=Path, default=Path('scene/data/uniri_seastate'))
    ap.add_argument('--out', type=Path, default=Path('scene_export'))
    ap.add_argument('--calib-count', type=int, default=512)
    ap.add_argument('--check-images', type=int, default=8)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    water = WaterNoUpsample(load_weights(load_script('train_water').WaterNet(), a.water)).eval()
    horizon = load_weights(load_script('train_horizon').HorizonNet(), a.horizon)
    seastate = load_weights(load_script('train_seastate_v2').build_model(), a.seastate)
    scene_imgs, sea_imgs = sample_images(a.lars, a.uniri, a.check_images)

    meta = {
        'preprocess': {'scene': 'letterbox to 640x384 (keep aspect, pad 114), RGB',
                       'seastate': '331x331 water patch of the original frame, centre 224x224 crop, RGB',
                       'normalisation': 'ImageNet mean/std; on chip for the Hailo (feed uint8 0-255)'},
        'water': export(water, 'water', (sc.H, sc.W), a.out, 'seg_logits', scene_imgs),
        'horizon': export(horizon, 'horizon', (sc.H, sc.W), a.out, 'horizon', scene_imgs),
        'seastate': export(seastate, 'seastate', (sc.SEA_SIZE, sc.SEA_SIZE), a.out, 'beaufort_logits', sea_imgs),
        'outputs': {'water': '(3,48,80) logits obstacle/water/sky at 1/8 size; upsample bilinear then argmax',
                    'horizon': '(2,) y_left, y_right of the water edge / 384 on the 640x384 canvas',
                    'seastate': '(8,) Beaufort 1-8 logits; calm=1-2, moderate=3-4, rough=5-8'},
        'checkpoints': {'water': str(a.water), 'horizon': str(a.horizon), 'seastate': str(a.seastate)},
    }
    write_calib(a.lars, a.uniri, a.out, a.calib_count)
    json.dump(meta, open(a.out / 'scene_meta.json', 'w'), indent=1)
    print('wrote', a.out / 'scene_meta.json')


if __name__ == '__main__':
    main()
