#!/usr/bin/env python3
"""Ground-truth accuracy of the 3 scene models, on the Hailo (INT8 HEFs) or with ONNX Runtime (FP32).

Uses the same labelled test sets and the same metrics as the laptop training scripts:
  water    LaRS test, 293 images 640x384 + masks   -> IoU obstacle / water / sky, mean IoU   (train_water.py)
  horizon  LaRS test, 232 images with a clean line -> mean error in px, share within 10 px   (train_horizon.py)
  seastate UNIRI test, 2,022 patches (224 crops)   -> accuracy, balanced accuracy, per class,
                                                      confusion, Beaufort error; single and mirror-averaged
                                                      (train_seastate_v2.py)

  python3 eval_scene_models.py --backend hailo --models /path/to/folder/with/hefs
  python3 eval_scene_models.py --backend onnx  --models /path/to/folder/with/onnx    # FP32 check (slower)

Prints each result next to the laptop FP32 reference (reference_fp32.json) and writes eval_<backend>.json.
Input to the models is uint8 RGB; on the Hailo the ImageNet normalisation runs on chip.
"""
import argparse
import contextlib
import json
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
W, H = 640, 384
SEG = ['obstacle', 'water', 'sky']
SEA = ['calm', 'moderate', 'rough']
GROUP = np.array([0, 0, 1, 1, 2, 2, 2, 2])          # Beaufort 1..8 -> class
MEAN = np.array([0.485, 0.456, 0.406], np.float32)
STD = np.array([0.229, 0.224, 0.225], np.float32)


class HailoModels:
    """Several HEFs on one Hailo-8L via the HailoRT round-robin scheduler (same as the Pi's scene_pi.py)."""
    def __init__(self, hefs):
        from hailo_platform import (HEF, VDevice, HailoSchedulingAlgorithm, ConfigureParams, HailoStreamInterface,
                                    InputVStreamParams, OutputVStreamParams, FormatType, InferVStreams)
        self.stack = contextlib.ExitStack()
        p = VDevice.create_params()
        p.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        vdev = self.stack.enter_context(VDevice(p))
        self.pipes, self.inputs = {}, {}
        for name, path in hefs.items():
            hef = HEF(str(path))
            ng = vdev.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
            ip = InputVStreamParams.make(ng, format_type=FormatType.UINT8)
            op = OutputVStreamParams.make(ng, format_type=FormatType.FLOAT32)
            self.pipes[name] = self.stack.enter_context(InferVStreams(ng, ip, op))   # scheduler: no activate()
            self.inputs[name] = hef.get_input_vstream_infos()[0].name

    def __call__(self, name, rgb):
        out = self.pipes[name].infer({self.inputs[name]: np.ascontiguousarray(rgb[None])})
        return np.asarray(next(iter(out.values())), np.float32)

    def close(self):
        self.stack.close()


class OnnxModels:
    def __init__(self, paths):
        import onnxruntime as ort
        self.sessions = {n: ort.InferenceSession(str(p), providers=['CPUExecutionProvider']) for n, p in paths.items()}

    def __call__(self, name, rgb):
        x = ((rgb.astype(np.float32) / 255.0 - MEAN) / STD).transpose(2, 0, 1)[None]
        return self.sessions[name].run(None, {'images': np.ascontiguousarray(x)})[0]

    def close(self):
        pass


def channels_first(t, c):
    t = np.asarray(t, np.float32).squeeze()
    return t if t.shape[0] == c else np.moveaxis(t, -1, 0)


def eval_water(run, data, limit):
    recs = [r for r in json.load(open(data / 'lars_test.json')) if r['mask']][:limit or None]
    inter, union = np.zeros(3), np.zeros(3)
    for r in recs:
        rgb = cv2.cvtColor(cv2.imread(str(data / r['image'])), cv2.COLOR_BGR2RGB)
        truth = cv2.imread(str(data / r['mask']), cv2.IMREAD_UNCHANGED)
        logits = channels_first(run('water', rgb), 3)                       # (3, 48, 80)
        up = np.stack([cv2.resize(c, (W, H), interpolation=cv2.INTER_LINEAR) for c in logits])
        pred = up.argmax(0)
        valid = truth != 255
        for c in range(3):
            p, t = (pred == c) & valid, (truth == c) & valid
            inter[c] += (p & t).sum()
            union[c] += (p | t).sum()
    iou = inter / np.maximum(union, 1)
    return {'images': len(recs), 'mean_iou': float(iou.mean()), 'iou': dict(zip(SEG, iou.tolist()))}


def eval_horizon(run, data, limit):
    recs = [r for r in json.load(open(data / 'lars_test.json')) if r['horizon_ok']][:limit or None]
    errors = []
    for r in recs:
        rgb = cv2.cvtColor(cv2.imread(str(data / r['image'])), cv2.COLOR_BGR2RGB)
        out = np.asarray(run('horizon', rgb), np.float32).reshape(-1)[:2] * H
        errors.append(float(np.abs(out - np.array([r['y_left'], r['y_right']])).mean()))
    e = np.array(errors)
    return {'images': len(recs), 'mean_error_px': float(e.mean()), 'within_10px': float((e <= 10).mean())}


def eval_seastate(run, data, limit):
    recs = json.load(open(data / 'seastate_test.json'))[:limit or None]
    result = {}
    logits_single, logits_flip, levels = [], [], []
    for r in recs:
        rgb = cv2.cvtColor(cv2.imread(str(data / r['image'])), cv2.COLOR_BGR2RGB)
        a = np.asarray(run('seastate', rgb), np.float32).reshape(-1)[:8]
        b = np.asarray(run('seastate', np.ascontiguousarray(rgb[:, ::-1])), np.float32).reshape(-1)[:8]
        logits_single.append(a)
        logits_flip.append((a + b) / 2)
        levels.append(r['beaufort'])
    levels = np.array(levels)
    for name, z in (('single', np.array(logits_single)), ('flip_tta', np.array(logits_flip))):
        p = np.exp(z - z.max(1, keepdims=True))
        p /= p.sum(1, keepdims=True)
        cls_p = np.stack([p[:, 0:2].sum(1), p[:, 2:4].sum(1), p[:, 4:8].sum(1)], 1)
        pred, true = cls_p.argmax(1), GROUP[levels - 1]
        conf = np.zeros((3, 3), int)
        for t, q in zip(true, pred):
            conf[t, q] += 1
        per = conf.diagonal() / conf.sum(1).clip(min=1)
        result[name] = {'images': len(recs), 'accuracy': float(conf.trace() / conf.sum()),
                        'balanced_accuracy': float(per.mean()), 'per_class': dict(zip(SEA, per.tolist())),
                        'confusion': conf.tolist(),
                        'beaufort_mean_abs_error': float(np.abs((p * np.arange(1, 9)).sum(1) - levels).mean())}
    return result


def show(name, got, ref):
    def row(label, key_path, fmt='{:.3f}'):
        g, r = got, ref
        for k in key_path:
            g = g[k]
            r = r[k] if r is not None else None
        delta = '' if r is None else f'   laptop FP32 {fmt.format(r)}   diff {g - r:+.3f}'
        print(f'  {label:28s} {fmt.format(g)}{delta}')
    print(f'\n{name}')
    if name == 'water':
        row('mean IoU', ['mean_iou'])
        for c in SEG:
            row(f'IoU {c}', ['iou', c])
    elif name == 'horizon':
        row('mean error (px)', ['mean_error_px'], '{:.2f}')
        row('within 10 px', ['within_10px'])
    else:
        for mode in ('single', 'flip_tta'):
            row(f'{mode}: accuracy', [mode, 'accuracy'])
            row(f'{mode}: balanced accuracy', [mode, 'balanced_accuracy'])
            for c in SEA:
                row(f'{mode}: {c}', [mode, 'per_class', c])
            row(f'{mode}: Beaufort error', [mode, 'beaufort_mean_abs_error'], '{:.2f}')
        print('  confusion (single, rows = true calm/moderate/rough):', got['single']['confusion'])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backend', choices=['hailo', 'onnx'], required=True)
    ap.add_argument('--models', type=Path, required=True, help='folder with water / horizon / seastate .hef or .onnx')
    ap.add_argument('--data', type=Path, default=HERE / 'data')
    ap.add_argument('--only', choices=['water', 'horizon', 'seastate'], action='append')
    ap.add_argument('--limit', type=int, default=0, help='first N images per test set (0 = all)')
    a = ap.parse_args()
    names = a.only or ['water', 'horizon', 'seastate']
    ext = 'hef' if a.backend == 'hailo' else 'onnx'
    paths = {n: a.models / f'{n}.{ext}' for n in names}
    missing = [str(p) for p in paths.values() if not p.exists()]
    if missing:
        raise SystemExit('missing model files: ' + ', '.join(missing))
    run = HailoModels(paths) if a.backend == 'hailo' else OnnxModels(paths)
    ref_path = HERE / 'reference_fp32.json'
    ref = json.load(open(ref_path)) if ref_path.exists() else {}
    results = {'backend': a.backend, 'models': {n: str(p) for n, p in paths.items()}}
    try:
        for n in names:
            t = time.perf_counter()
            results[n] = {'water': eval_water, 'horizon': eval_horizon, 'seastate': eval_seastate}[n](run, a.data, a.limit)
            print(f'[{n}] done in {time.perf_counter() - t:.0f} s', flush=True)
            show(n, results[n], ref.get(n) if not a.limit else None)
    finally:
        run.close()
    out = HERE / f'eval_{a.backend}.json'
    json.dump(results, open(out, 'w'), indent=1)
    print('\nwrote', out)


if __name__ == '__main__':
    main()
