# Maritime6 960 on Hailo-8L: compile, Pi baseline, RTL vectors

Model: YOLOv8s, 6 classes, 960x960, trained on `maritime6_demo_d2` (test mAP50 0.840 in Ultralytics).

## What was checked on the laptop (2026-10-30)

| Check | Result |
|---|---|
| Exported ONNX on d2 val (Ultralytics) | mAP50 0.827 (trained model: 0.820) |
| Cut ONNX (2 raw outputs) on d2 test, `maritime_post.py` | **mAP50 0.807, mAP50-95 0.550**: the float reference for the HEF |
| `pi_baseline.py`, `dump_vectors.py` with a stand-in Hailo library | run end to end |
| `golden_model.py` on the stand-in dump (960, 6 classes, 18,900 anchors) | 12/12 frames: fixed point matches float |

The 0.807 vs 0.840 gap is the scorer: `maritime_post` keeps one class per box (like the deployed
code and the RTL), while Ultralytics val keeps several. Compare the Pi result with **0.807**.

## Step 2: compile (machine with the Hailo Dataflow Compiler 3.33, x86 Linux or WSL2)

```bash
tar -xzf maritime6_hailo_export.tar.gz          # hailo_export/maritime6_960.onnx + calib/
python compile_maritime6_960.py --onnx hailo_export/maritime6_960.onnx --calib hailo_export/calib
```
On a machine without a GPU add `--opt-level 1`. Expected outputs: `maritime6_960.hef`,
`compile_report.json`. The report must show **2 outputs**, shapes 18,900 x 64 and 18,900 x 6.
If the parser refuses the cut, send the error text back; fallback `--layout split6` changes the RTL contract.

Output contract: input 960x960x3 uint8 RGB letterboxed (pad 114), /255 on chip;
`bbox` 18,900 x 64 DFL logits, `score` 18,900 x 6 sigmoid probabilities; no NMS on chip.
Anchor order: stride 8 (120x120), 16 (60x60), 32 (30x30), row-major.

## Step 3: Pi baseline (Raspberry Pi 5 + Hailo-8L, the environment where HailoRT works)

```bash
tar -xzf maritime6_pi_bundle.tar.gz && cd maritime6_pi
hailortcli fw-control identify
hailortcli run maritime6_960.hef                    # hardware-only FPS
python3 pi_baseline.py --hef maritime6_960.hef --images test/images --labels test/labels
```
Prints mAP50 / mAP50-95 per class and the median time per stage (letterbox, Hailo, CPU post-processing).
Saves `pi_baseline/pi_baseline_results.json` and 12 overlay images.

## Step 4: vectors for the RTL team

```bash
python3 dump_vectors.py --hef maritime6_960.hef --images test/images --labels test/labels --frames 30 --out fpga_vectors
tar -czf fpga_vectors.tar.gz fpga_vectors
```
Bring `fpga_vectors.tar.gz` back; `golden_model.py --vectors fpga_vectors --conf 0.25 --iou 0.45`
then builds `rtl_params.vh`, `dfl_exp_lut.mem` and the three checkpoints per frame from the REAL quantization.
