# Handoff: compile the 3 scene models to Hailo-8L HEFs (Linux PC with the DFC)

For the Claude Code session on the Linux PC that already compiled `maritime6_960.hef`. Same user
(short plain answers, action over analysis, no unrequested experiments), same DFC 3.33 environment.

## Task

```bash
tar -xzf scene_hailo_export.tar.gz
python compile_scene_models.py --export scene_export          # --opt-level 1 if no GPU
```
Expected: `scene_hefs/water.hef`, `horizon.hef`, `seastate.hef`, `compile_report.json`.
One model at a time: `--only water` (or horizon / seastate); the report keeps earlier entries.

## The models (all ResNet18, ONNX opset 13, input name `images`, ImageNet-normalised float input)

| Model | Input | ONNX output | Layers after the ResNet18 trunk |
|---|---|---|---|
| water | 1x3x384x640 | `seg_logits` 1x3x48x80 | 1x1 convs, nearest x2 Resize + Add (twice), 3x3 conv+BN+ReLU, 1x1 conv |
| horizon | 1x3x384x640 | `horizon` 1x2 | 1x1 conv 512->32 + ReLU, Flatten (32x12x20=7680), Gemm 7680->256, ReLU, Gemm 256->2 |
| seastate | 1x3x224x224 | `beaufort_logits` 1x8 | GlobalAveragePool, Flatten, Gemm 512->8 |

Model script: `normalization([123.675,116.28,103.53],[58.395,57.12,57.375])` (ImageNet on chip, the Pi feeds
uint8 0-255 RGB NHWC) + `model_optimization_flavor(optimization_level=2, compression_level=0)`.
Calibration: `scene_export/calib/scene` (512 letterboxed 640x384 LaRS frames) and `calib/seastate`
(512 UNIRI 224x224 crops).

## Output contract the Pi relies on (`scene_common.py`)

- water: 3 channels at 48x80 in any layout (NHWC or NCHW; `channels_first` handles both), float32 logits.
- horizon: 2 floats (y_left, y_right) / 384. Any shape that flattens to 2 values.
- seastate: 8 floats (Beaufort 1-8 logits). Any shape that flattens to 8 values.
The Pi reads outputs as FLOAT32 (dequantised by HailoRT) and uses the first output of each HEF.

## Likely trouble and what to try

- Parser asks for end nodes: pass the last node before the output (the final Conv for water, the last
  Gemm for horizon and seastate) via `end_node_names`.
- horizon's Flatten -> Gemm 7680->256 is the most unusual part. If the parser or allocator refuses it,
  tell the user before changing the model; an option is re-exporting horizon with the Flatten+Gemm written
  as a 12x20 "valid" convolution (same weights, reshaped), done on the laptop.
- Accuracy check after compile: run the quantized HAR in the SDK emulator on ~20 calib images and compare
  with ONNX Runtime (water: pixel agreement of argmax; horizon: mean px error x384; seastate: same argmax).
  Report numbers, do not tune beyond opt-level unless the user asks.
