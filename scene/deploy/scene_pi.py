#!/usr/bin/env python3
"""Run the 3 scene models (water mask, horizon line, sea state) on the Pi, optionally with the detector.

Two backends for the scene models:
  --backend hailo : water.hef, horizon.hef, seastate.hef on the Hailo-8L (needs compile_scene_models.py first)
  --backend onnx  : water.onnx, horizon.onnx, seastate.onnx on the Pi CPU with ONNX Runtime (works today, slower)
The detector (--det-hef maritime6_960.hef) always runs on the Hailo. All Hailo models share one device
through the HailoRT round-robin scheduler.

  python3 scene_pi.py --backend onnx  --models . --images ../test/images --frames 20
  python3 scene_pi.py --backend hailo --models . --images ../test/images --det-hef ../maritime6_960.hef --gate
  python3 scene_pi.py --backend hailo --models . --video 0 --det-hef ../maritime6_960.hef --gate   # USB camera

Writes scene_results/results.jsonl (one line per frame), summary.json (timing per model) and overlays:
water blue, sky grey, obstacles red, horizon yellow, sea-state patch white, kept boxes green,
boxes removed by the horizon gate thin red.
"""
import argparse
import json
import time
from pathlib import Path

import cv2
import numpy as np

import scene_common as sc


class OnnxModel:
    def __init__(self, path):
        import onnxruntime as ort
        self.session = ort.InferenceSession(str(path), providers=['CPUExecutionProvider'])

    def __call__(self, rgb):
        return self.session.run(None, {'images': sc.to_onnx_input(rgb)})[0]


class HailoDevice:
    """One VDevice, several HEFs, HailoRT scheduler (no manual activation)."""
    def __init__(self):
        from hailo_platform import VDevice, HailoSchedulingAlgorithm
        params = VDevice.create_params()
        params.scheduling_algorithm = HailoSchedulingAlgorithm.ROUND_ROBIN
        self.device = VDevice(params)
        self.pipes = []

    def load(self, hef_path):
        from hailo_platform import (HEF, ConfigureParams, HailoStreamInterface, InputVStreamParams,
                                    OutputVStreamParams, FormatType, InferVStreams)
        hef = HEF(str(hef_path))
        group = self.device.configure(hef, ConfigureParams.create_from_hef(hef, interface=HailoStreamInterface.PCIe))[0]
        pipe = InferVStreams(group, InputVStreamParams.make(group, format_type=FormatType.UINT8),
                             OutputVStreamParams.make(group, format_type=FormatType.FLOAT32))
        pipe.__enter__()
        self.pipes.append(pipe)
        in_name = hef.get_input_vstream_infos()[0].name
        out_names = [o.name for o in hef.get_output_vstream_infos()]

        def run(rgb, all_outputs=False):
            res = pipe.infer({in_name: np.ascontiguousarray(rgb[None])})
            return res if all_outputs else res[out_names[0]]
        return run

    def close(self):
        for pipe in self.pipes:
            pipe.__exit__(None, None, None)
        self.device.release()


def frames(a):
    if a.video is not None:
        cap = cv2.VideoCapture(int(a.video) if a.video.isdigit() else a.video)
        k = 0
        while not a.frames or k < a.frames:
            ok, img = cap.read()
            if not ok:
                break
            yield f'frame_{k:05d}', img
            k += 1
        cap.release()
        return
    files = sorted(p for p in a.images.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))
    for p in files[:a.frames] if a.frames else files:
        yield p.stem, cv2.imread(str(p))


def horizon_agreement(mask, line, tol=30):
    """Share of columns whose first water pixel (from the top) is within tol px of the predicted line.
    Low agreement = no clean water edge (canal, harbour) -> do not trust the line for gating."""
    water = mask == 1
    cols = np.nonzero(water.any(0))[0]
    if len(cols) < sc.W * 0.2:
        return 0.0
    top = water[:, cols].argmax(0)
    y_left, y_right = line
    pred = y_left + (y_right - y_left) * cols / (sc.W - 1)
    return float((np.abs(top - pred) <= tol).mean())


def stats(x):
    x = np.asarray(x, float)
    return {'n': int(x.size), 'median_ms': float(np.median(x)), 'p95_ms': float(np.percentile(x, 95))} if x.size else None


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--backend', choices=['hailo', 'onnx'], required=True)
    ap.add_argument('--models', type=Path, default=Path('.'), help='folder with water/horizon/seastate .hef or .onnx')
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--images', type=Path)
    src.add_argument('--video', help='video file, or a camera number such as 0')
    ap.add_argument('--frames', type=int, default=0, help='0 = all')
    ap.add_argument('--det-hef', type=Path, help='also run the detector (maritime6_960.hef) on the Hailo')
    ap.add_argument('--conf', type=float, default=0.25)
    ap.add_argument('--iou', type=float, default=0.45)
    ap.add_argument('--gate', action='store_true', help='drop detections whose bottom is above the horizon')
    ap.add_argument('--gate-margin', type=float, default=20, help='pixels of slack above the horizon')
    ap.add_argument('--smooth', type=float, default=0.0, help='0-1: average sea state over frames (video)')
    ap.add_argument('--no-flip', action='store_true', help='run the sea-state model once instead of twice')
    ap.add_argument('--overlays', type=int, default=20)
    ap.add_argument('--out', type=Path, default=Path('scene_results'))
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)

    hailo = HailoDevice() if a.backend == 'hailo' or a.det_hef else None
    ext = 'hef' if a.backend == 'hailo' else 'onnx'
    load = hailo.load if a.backend == 'hailo' else OnnxModel
    models = {name: load(a.models / f'{name}.{ext}') for name in ('water', 'horizon', 'seastate')}
    detector = None
    if a.det_hef:
        import maritime_post as mp
        detector = hailo.load(a.det_hef)
    print(f'scene models: {ext} from {a.models}' + (f'   detector: {a.det_hef}' if a.det_hef else ''), flush=True)

    times = {k: [] for k in ('letterbox', 'water', 'horizon', 'seastate', 'detector', 'total')}
    smooth_probs = None
    results = open(a.out / 'results.jsonl', 'w')
    n = 0
    for name, img in frames(a):
        h, w = img.shape[:2]
        t0 = time.perf_counter()
        rgb, r, pad = sc.letterbox(img)
        t1 = time.perf_counter()
        mask = sc.water_mask(models['water'](rgb))
        t2 = time.perf_counter()
        line, y_at = sc.horizon_line(models['horizon'](rgb), r, pad)
        t3 = time.perf_counter()
        patch, patch_xy, water_frac = sc.sea_patch(img, mask, r, pad, y_at)
        sea = None
        if patch is not None:
            logits = np.asarray(models['seastate'](patch), np.float32).reshape(-1)
            if not a.no_flip:   # average with the mirror image (test accuracy 0.770 -> 0.795)
                logits = (logits + np.asarray(models['seastate'](np.ascontiguousarray(patch[:, ::-1])),
                                              np.float32).reshape(-1)) / 2
            probs, level = sc.sea_state(logits)
            if a.smooth and smooth_probs is not None:
                probs = a.smooth * smooth_probs + (1 - a.smooth) * probs
            smooth_probs = probs
            sea = (probs, level)
        t4 = time.perf_counter()
        boxes, labels, dropped, det = np.zeros((0, 4)), [], [], []
        if detector:
            drgb, dr, dpad = mp.letterbox(img)
            out = detector(drgb, all_outputs=True)
            sizes = {k: v.size for k, v in out.items()}
            bbox = mp.to_anchor_major(out[next(k for k, s in sizes.items() if s == mp.N_ANCHORS * 64)], 64)
            score = mp.to_anchor_major(out[next(k for k, s in sizes.items() if s == mp.N_ANCHORS * 6)], 6)
            b, s, c = mp.decode(bbox, score, a.conf, a.iou)
            b = mp.unletterbox(b, dr, dpad, w, h)
            agree = horizon_agreement(mask, line)
            keep = sc.gate_detections(b, y_at, a.gate_margin) if a.gate and agree >= 0.5 else np.ones(len(b), bool)
            boxes = b[keep]
            labels = [f'{mp.NAMES[ci]} {si:.2f}' for ci, si in zip(c[keep], s[keep])]
            dropped = b[~keep]
            det = [{'cls': mp.NAMES[ci], 'score': round(float(si), 3), 'box': [round(float(v), 1) for v in bi],
                    'kept': bool(k)} for bi, si, ci, k in zip(b, s, c, keep)]
        t5 = time.perf_counter()
        for key, dt in zip(times, (t1 - t0, t2 - t1, t3 - t2, t4 - t3, t5 - t4, t5 - t0)):
            if key != 'detector' or detector:
                times[key].append(dt * 1e3)
        row = {'frame': name,
               'water_fraction': round(float((mask == 1).mean()), 3),
               'horizon_y_left_right_canvas': [round(v, 1) for v in line],
               'horizon_agreement': round(horizon_agreement(mask, line), 3),
               'sea_state': None if sea is None else sc.SEA_CLASSES[int(np.argmax(sea[0]))],
               'sea_probs': None if sea is None else [round(float(p), 3) for p in sea[0]],
               'beaufort_estimate': None if sea is None else round(sea[1], 2),
               'sea_patch_water_fraction': round(water_frac, 3),
               'detections': det}
        results.write(json.dumps(row) + '\n')
        if n < a.overlays:
            view = sc.overlay(img, mask, r, pad, y_at, sea, patch_xy if patch is not None else None,
                              boxes, labels, dropped)
            cv2.imwrite(str(a.out / f'{name}_scene.jpg'), view)
        msg = f"{name}: water {row['water_fraction']:.2f}  horizon agree {row['horizon_agreement']:.2f}  "
        msg += f"sea {row['sea_state']} (Beaufort ~{row['beaufort_estimate']})" if sea else 'sea: no water patch'
        if detector:
            msg += f'  dets {int(sum(d["kept"] for d in det))} kept / {len(det)}'
        print(msg + f'  {times["total"][-1]:.0f} ms', flush=True)
        n += 1
    results.close()
    summary = {'backend': a.backend, 'frames': n, 'timing': {k: stats(v) for k, v in times.items()}}
    if times['total']:
        summary['fps'] = 1000.0 / summary['timing']['total']['median_ms']
    json.dump(summary, open(a.out / 'summary.json', 'w'), indent=1)
    print(json.dumps(summary, indent=1))
    if hailo:
        hailo.close()


if __name__ == '__main__':
    main()
