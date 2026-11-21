# Scene models: INT8 accuracy on the Hailo-8L against ground truth

Fills the gap in the 2026-11-16 results ("no labelled water / horizon / sea-state test images on the Pi").
Runs `water.hef`, `horizon.hef`, `seastate.hef` on the same labelled test sets the laptop used, with the
same metrics, and prints every number next to the laptop FP32 result.

| Model | Test set (in `data/`) | Metric | Laptop FP32 (`reference_fp32.json`) |
|---|---|---|---|
| water | LaRS test, 293 images + masks (640x384) | IoU obstacle / water / sky, mean IoU | mIoU **0.955** (0.932 / 0.962 / 0.973) |
| horizon | LaRS test, 232 images with a clean water line | mean error, share within 10 px | **7.84 px**, **80.2 %** |
| seastate | UNIRI test, 2,022 patches (224x224 centre crops) | accuracy, balanced accuracy, per class, Beaufort error | **0.770** single, **0.795** mirror-averaged |

The FP32 reference was made with this script and the ONNX files whose sha256 are in
`scene_compile_report.json` (fe4d04e8 water, 2f40d1f0 horizon, 71e7c1bd seastate), i.e. exactly the models
compiled into the HEFs. Images are fed as uint8 RGB; the ImageNet normalisation runs on chip.

## Run on the Pi

```bash
tar -xzf scene_eval_kit.tar.gz && cd scene_eval
python3 eval_scene_models.py --backend hailo --models /media/sarva963/Saravanan/maritime6/scene_models
```
(`--models` = the folder that holds the three `.hef` files.) Takes a few minutes; 2,022 x 2 sea-state runs
is the longest part. Quick try first: add `--limit 20`. One model only: `--only seastate`.

Output: a table per model with `laptop FP32 ... diff ...`, and `eval_hailo.json` with everything (also the
sea-state confusion matrix). Bring `eval_hailo.json` back.

## Reading the result

- `diff` is the INT8 cost. Water and horizon should be close (the 20-image check gave 99.0 % pixel
  agreement and 0.5 px). Sea state is the one to watch: the 20-image check had a logit moving by 16.3.
- If sea state drops a lot (more than ~0.03 accuracy), the usual fixes are recompiling seastate with
  `--opt-level 2` on a GPU machine, or more calibration images; decide after seeing the numbers.
