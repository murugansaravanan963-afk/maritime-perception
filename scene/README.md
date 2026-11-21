# Scene models: water segmentation, horizon, sea state

Three models that sit next to the Maritime6 detector (proposal: 4-model system).

| Model | Input | Output | Data |
|---|---|---|---|
| Water segmentation | full frame, letterboxed 640x384 | obstacle / water / sky per pixel | LaRS masks (`data/scene_lars_v1`) |
| Horizon | full frame, 640x384 | water-edge line (y at left and right edge) | lines fitted to the LaRS water masks |
| Sea state | water patch below the horizon | calm (Bft 1-2) / moderate (3-4) / rough (5-8) | UNIRI-SeaState real ship-camera patches (`data/uniri_seastate`) |

## Dataset `data/scene_lars_v1` (water + horizon)

- Built by `build_scene_data.py` (letterbox, masks, horizon lines), split by `split_scene_data.py`.
- Mask values: 0 obstacle, 1 water, 2 sky, 255 ignore (also the letterbox padding).
- Horizon line: robust fit through the top water pixel of every column; columns where a boat
  sits on the water edge are ignored (LaRS panoptic masks). `line_ok` = the edge really is a line
  (narrow canals with V-shaped banks are not; the horizon model trains on `line_ok` only).
- Splits (`split_summary.json`):
  - train 2,312 / val 198 (LaRS official val) / **test 293** (our own) / lars_test 1,203 (no public masks, unused).
  - Test = whole *sessions* held out from LaRS train: recordings are joined into one session when
    any of their frames look nearly identical (thumbnail correlation >= 0.93), because sequence
    names are not reliable (davimar_seq_16 and _17 are the same place and day).
  - MaSTr1325 is train-only (its ids are not in time order; the same day appears under far-apart ids).
  - Check: 0 sessions shared between train and test; highest train/test thumbnail correlation 0.929.
- Images are checked: all 4,006 open, 640x384; masks contain only 0/1/2/255.

## Dataset `data/uniri_seastate` (sea state)

`fetch_uniri.py` downloads only the real RGB images (R750 train + RGB test, ~1.3 GB of the 6.8 GB zip).
UNIRI's test videos share none with its training videos. Val = 15% of the training videos per class.
Result (`summary.json`): all 8,014 images open, 331x331 RGB.

| Split | Images | Videos | calm | moderate | rough |
|---|---|---|---|---|---|
| train | 5,036 | 99 | 1,257 | 1,268 | 2,511 |
| val | 956 | 22 | 243 | 232 | 481 |
| test | 2,022 | 90 | 450 | 600 | 972 |

0 videos shared between any two splits. Val has no Beaufort 7-8 (only 2 training videos each, kept
in train); the test set has all eight.
Licence CC BY 4.0: cite Paulin, Pobar, Ivasic-Kos, Vorkapic, IEEE OJ-ITS 7 (2026), doi 10.1109/OJITS.2025.3650057.

## Reference training script

`train_scene.py` + `scene_models.py` (tested on the laptop). The student version is written in lessons.
