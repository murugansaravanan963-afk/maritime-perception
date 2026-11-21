#!/usr/bin/env python3
"""Compile the 3 scene ONNX models (water, horizon, seastate) to Hailo-8L HEFs.

Run inside the Hailo Dataflow Compiler environment (DFC 3.33, x86 Linux or WSL2) — the same place
compile_maritime6_960.py ran — from the folder that holds scene_export/:
    python compile_scene_models.py --export scene_export
    python compile_scene_models.py --export scene_export --only seastate      # one model
On a machine without a GPU add --opt-level 1.

Input contract on the Pi: uint8 RGB, NHWC. ImageNet normalisation runs on chip, so do NOT normalise.
  water, horizon : 384x640x3, letterboxed exactly like scene_common.letterbox (pad 114)
  seastate       : 224x224x3 water patch (scene_common.sea_patch)
"""
import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np

MODELS = {   # name: (input height, width, calibration folder)
    'water': (384, 640, 'scene'),
    'horizon': (384, 640, 'scene'),
    'seastate': (224, 224, 'seastate'),
}
NORMALISATION = 'normalization1 = normalization([123.675, 116.28, 103.53], [58.395, 57.12, 57.375])'


def calibration(folder, h, w, count):
    files = sorted(p for p in folder.iterdir() if p.suffix.lower() in ('.jpg', '.jpeg', '.png'))[:count]
    if not files:
        raise SystemExit(f'no calibration images in {folder}')
    imgs = []
    for p in files:
        img = cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB)
        if img.shape[:2] != (h, w):
            raise SystemExit(f'{p} is {img.shape[:2]}, expected {(h, w)}')
        imgs.append(img)
    return np.stack(imgs)


def compile_one(name, export_dir, out_dir, opt_level, calib_count):
    from hailo_sdk_client import ClientRunner
    h, w, calib_folder = MODELS[name]
    onnx_path = export_dir / f'{name}.onnx'
    runner = ClientRunner(hw_arch='hailo8l')
    runner.translate_onnx_model(str(onnx_path), name, start_node_names=['images'],
                                net_input_shapes={'images': [1, 3, h, w]})
    runner.save_har(str(out_dir / f'{name}_parsed.har'))
    hn = runner.get_hn()
    hn = json.loads(hn) if isinstance(hn, str) else hn
    outs = hn['net_params']['output_layers_order']
    shapes = {n: hn['layers'][n]['output_shapes'][0] for n in outs}
    print(f'[{name}] parsed outputs: {json.dumps(shapes)}', flush=True)
    script = NORMALISATION + '\n' + f'model_optimization_flavor(optimization_level={opt_level}, compression_level=0)\n'
    (out_dir / f'{name}.alls').write_text(script)
    runner.load_model_script(script)
    calib = calibration(export_dir / 'calib' / calib_folder, h, w, calib_count)
    print(f'[{name}] calibration {calib.shape} {calib.dtype}', flush=True)
    runner.optimize(calib)
    runner.save_har(str(out_dir / f'{name}_quantized.har'))
    hef = runner.compile()
    (out_dir / f'{name}.hef').write_bytes(hef)
    return {'hef': f'{name}.hef', 'hef_sha256': hashlib.sha256(hef).hexdigest(),
            'onnx_sha256': hashlib.sha256(onnx_path.read_bytes()).hexdigest(), 'output_layers': outs,
            'output_shapes': shapes, 'alls': script, 'calibration_images': int(len(calib)),
            'input': f'{h}x{w}x3 uint8 RGB; ImageNet normalisation on chip'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--export', type=Path, default=Path('scene_export'))
    ap.add_argument('--out', type=Path, default=Path('scene_hefs'))
    ap.add_argument('--only', choices=list(MODELS), action='append')
    ap.add_argument('--opt-level', type=int, default=2, help='2 is best with a GPU; use 1 on a CPU-only machine')
    ap.add_argument('--calib-count', type=int, default=512)
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=True)
    report_path = a.out / 'compile_report.json'
    report = json.loads(report_path.read_text()) if report_path.exists() else {}
    for name in a.only or list(MODELS):
        report[name] = compile_one(name, a.export, a.out, a.opt_level, a.calib_count)
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        print(f'[{name}] wrote {a.out / (name + ".hef")}', flush=True)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
