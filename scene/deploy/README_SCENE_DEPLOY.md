# Deploying the 3 scene models (water, horizon, sea state) on the Pi

```
 camera frame ─┬─ letterbox 640x384 ─┬─ water model ───► mask (obstacle / water / sky)
               │                     └─ horizon model ─► water-edge line ──► horizon gate on detections
               ├─ 331 px water patch below the horizon ─► sea-state model ─► calm / moderate / rough + Beaufort
               └─ letterbox 960 ───── detector (maritime6_960.hef) ─► boxes
```

All models are ResNet18-based and use only Conv / BN / ReLU / pooling / resize / dense layers, so they
compile for the Hailo-8L. Until the HEFs exist, the same Pi script runs the ONNX files on the Pi CPU.

| File | Where it runs | What it does |
|---|---|---|
| `export_scene_onnx.py` | laptop | `.pt` → `water.onnx`, `horizon.onnx`, `seastate.onnx` (checked against PyTorch) + calibration images |
| `compile_scene_models.py` | Linux PC with the Hailo DFC | ONNX → `water.hef`, `horizon.hef`, `seastate.hef` |
| `scene_pi.py` + `scene_common.py` | Pi | runs the 3 models (+ detector), horizon gate, overlays, timing |

## Step 1: laptop, get the water and horizon checkpoints from Sol

Download these two files from Sol and put them on the laptop:

| Sol | Laptop |
|---|---|
| `/scratch/smurug25/maritime-dataset/runs_scene/water/last.pt` | `D:\Projects\maritime-dataset\runs_scene\water\last.pt` |
| `/scratch/smurug25/maritime-dataset/runs_scene/horizon/last.pt` | `D:\Projects\maritime-dataset\runs_scene\horizon\last.pt` |

## Step 2: laptop, export

```bash
python scene/deploy/export_scene_onnx.py --water runs_scene/water/last.pt --horizon runs_scene/horizon/last.pt --seastate runs_scene/seastate_v2/best.pt --out scene_export
```
Each model prints `max |torch - onnx|`; it must be below 1e-3 (the script stops otherwise).

## Step 3: Linux PC with the Hailo Dataflow Compiler

Copy `scene_export/` and `compile_scene_models.py` there, then:
```bash
python compile_scene_models.py --export scene_export
```
(`--opt-level 1` without a GPU.) Output: `scene_hefs/water.hef`, `horizon.hef`, `seastate.hef`, `compile_report.json`.

## Step 4: Pi

One time: a Python environment that sees the system HailoRT and has ONNX Runtime:
```bash
python3 -m venv --system-site-packages ~/scene_env
~/scene_env/bin/pip install onnxruntime
```

CPU first (works before the HEFs exist):
```bash
cd ~/maritime6_pi/scene_pi
~/scene_env/bin/python scene_pi.py --backend onnx --models . --images ../test/images --frames 20
```

Hailo, all four models together, with the horizon gate:
```bash
~/scene_env/bin/python scene_pi.py --backend hailo --models . --images ../test/images --det-hef ../maritime6_960.hef --gate
```
USB camera: `--video 0` instead of `--images ...` (add `--smooth 0.8` to steady the sea state).

Output in `scene_results/`: `*_scene.jpg` overlays, `results.jsonl` (per frame), `summary.json` (median ms per model, FPS).

## What the outputs mean

- **water mask**: obstacle / water / sky per pixel. The Hailo returns it at 1/8 size (3x48x80); the Pi
  upsamples the logits (checked: 100% the same pixels as the full model).
- **horizon**: y of the water edge at the left and right of the 640x384 canvas. `horizon_agreement`
  = share of image columns where the water mask's top edge is within 30 px of the line. The gate is
  only applied when it is at least 0.5 (canals and harbours have no clean line).
- **horizon gate** (`--gate`): drops a detection whose bottom edge is more than `--gate-margin` px above
  the horizon (birds, clouds, things on shore). Dropped boxes are drawn thin red.
- **sea state**: calm / moderate / rough plus an estimated Beaufort level, from a 331x331 patch of the
  original frame that is at least 80% water and below the horizon (white square). No patch → no sea state.

## Known limits

- The sea-state model was trained on UNIRI patches (open sea, fixed camera). On our camera the patch
  scale and view angle differ, so treat it as a demo signal and smooth it over frames.
- `scene_pi.py --backend hailo` was tested on the laptop with a fake HailoRT that answers with the ONNX
  files (same input/output layouts). The real HailoRT scheduler path still needs its first run on the Pi.
