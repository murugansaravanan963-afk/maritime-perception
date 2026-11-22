# RTL package: YOLOv8 DFL + NMS for Maritime6 960 (2026-10-31)

From Saravanan Murugan (ASU). Test vectors come from the **real** Hailo-8L outputs of
`maritime6_960.hef` on the Raspberry Pi 5.

## What to build

The post-processing block in the FPGA: **DFL box decode → class argmax → class-aware NMS**.
Input per anchor: 64 uint8 DFL logits + 6 uint8 class scores. Output: up to 100 boxes
(`x1 y1 x2 y2` in 1/16 px, class, score, anchor index). Full algorithm in `SPEC.md` §3,
link and packet format in `SPI_INTERFACE.md`.

| Parameter | Value |
|---|---|
| Input size | 960 × 960 |
| Classes | 6 (powered_vessel, sailboat, small_craft, navigation_marker, person_in_water, other_obstacle) |
| Anchors | 18,900 (strides 8/16/32 → grids 120/60/30) |
| bbox quant | scale 0.1738095, zero point 171 (same for all three strides) |
| score quant | scale 1/255, zero point 0 (sigmoid already applied on the Hailo) |
| Threshold | conf 0.25 → `SCORE_Q_THR = 64` |
| NMS IoU | 0.45 → `9·union ≤ 20·inter` test, no divider |
| Caps | `MAX_CAND = 4096`, `MAX_DET = 100` |
| Link | SPI, **mode B**: the Pi sends only anchors with max score ≥ 64 (avg 11, max 42 per frame) |

All of this is in `rtl_vectors/rtl_params.vh` as `localparam`s.

## Start here

1. `SPEC.md`: the bit-accurate algorithm and the file formats.
2. `rtl_vectors/rtl_params.vh` and `rtl_vectors/dfl_exp_lut.mem`: include / `$readmemh` them.
3. `rtl_vectors/frame_0000 … frame_0029/`: testbench input (`in_bbox.hex`, `in_score.hex`)
   and the three expected checkpoints (`exp_dfl.hex`, `exp_decode.hex`, `exp_nms.hex`).
   `exp_nms.txt` shows the same detections in readable form.
4. `SPI_INTERFACE.md`: Pi ⇄ FPGA wiring and packets (v2, updated for 960).

## Pass criterion

Exact match with `exp_dfl.hex`, `exp_decode.hex` and `exp_nms.hex` on all 30 frames.
In mode B, compare the first two only for the anchors that were sent (`pass = 1`).
The Python reference already passes: `rtl_vectors/summary.txt` says
**30/30 frames: fixed-point matches float reference** (max box error 0.120 px).

## CPU numbers to compare against (one Pi 5 core, bit-exact C++)

| | Median |
|---|---|
| Full frame: DFL + decode + NMS on all 18,900 anchors | **2.910 ms** (DFL 2.637 ms = 91%) |
| Pi prefilter for mode B | 0.108 ms |
| CPU-only on the candidates (prefilter + DFL/decode/NMS) | 0.112 ms |

Details and the raw JSON in `cpu_baseline/`. Power per frame is not measured yet.

## Regenerating

```bash
python3 golden_model.py --vectors fpga_vectors --out rtl_vectors --conf 0.25 --iou 0.45
```
Needs only numpy. **Pass `--conf 0.25`**: the script's default (0.3) is from the older powerline model.
`fpga_vectors/` holds the raw Hailo uint8 outputs (`.npz`), `meta.json` (quant params) and the
source image of each frame. `dump_vectors.py` + `maritime_post.py` show how it was made on the Pi.

## Contents

| Path | What |
|---|---|
| `README.md` | this file |
| `SPEC.md` | algorithm + file formats (updated for 960 / 6 classes) |
| `SPI_INTERFACE.md` | SPI link, packets, wiring, what to measure (v2) |
| `golden_model.py` | bit-accurate Python model, generates `rtl_vectors/` |
| `rtl_vectors/` | `rtl_params.vh`, `dfl_exp_lut.mem`, `summary.txt`, 30 × `frame_XXXX/` |
| `fpga_vectors/` | raw Hailo outputs the vectors were made from |
| `bench/` | `cpu_bench.cpp`, `prefilter_bench.cpp`, `pi_efficiency.py`: CPU side of the CPU vs FPGA comparison |
| `cpu_baseline/` | measured one-core Pi 5 results for 960: `CPU_BASELINE.md` (summary) + raw JSON |
| `dump_vectors.py`, `maritime_post.py` | Pi-side reference scripts (need the Hailo) |

Note: comments inside `golden_model.py` still mention the powerline model; the math is the same,
and sizes / quant params are read from `fpga_vectors/meta.json`.
