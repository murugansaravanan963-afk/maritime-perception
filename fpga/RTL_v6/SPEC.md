# YOLOv8 DFL + NMS FPGA Block: RTL Hand-off Spec (Maritime6 960)

**Scope of the RTL work:** only the YOLOv8 post-processing (DFL box decode, score threshold, NMS) inside the FPGA. The neural network stays on the Hailo-8L. Pre-processing and display stay on the Raspberry Pi 5.

The block is parameterized: the same RTL serves any YOLOv8 detection model. All sizes come from `rtl_params.vh`, generated per model. **The target model for this package is Maritime6 960.**

| | **Maritime6 (this package)** | Powerline (earlier) |
|---|---|---|
| HEF | `maritime6_960.hef` (sha256 `849c06e9…46995`) | `powerline_yolov8n_8l.hef` |
| Model | YOLOv8s | YOLOv8n |
| Input W × H | **960 × 960** | 416 × 416 |
| Classes | **6**: 0 powered_vessel, 1 sailboat, 2 small_craft, 3 navigation_marker, 4 person_in_water, 5 other_obstacle | 20 |
| Grids (stride 8/16/32) | **120 / 60 / 30** | 52 / 26 / 13 |
| Anchors | **18,900** (14,400 + 3,600 + 900) | 3,549 |
| Input data / frame (uint8) | **1.32 MB** (18,900 × 70 B) | 298 KB |
| COORD / AREA / ANCHOR / CLASS bits | **14 / 28 / 15 / 3** | 13 / 26 / 12 / 5 |
| conf / NMS IoU | **0.25 / 0.45** → `SCORE_Q_THR = 64`, IoU = 9/20 | 0.3 / 0.45 |
| MAX_CAND / MAX_DET | 4096 / 100 | 1024 / 100 |

## 1. What runs where

| Stage | Today | Target |
|---|---|---|
| Backbone + head convs + class **sigmoid** | Hailo-8L | Hailo-8L (no change) |
| Join the 6 HEF outputs into one `bbox_q` / `score_q` stream | Pi CPU | Pi CPU (no change) |
| Score threshold (mode B prefilter, see `SPI_INTERFACE.md`) | Pi CPU | Pi CPU |
| DFL softmax + box decode | Pi CPU | **FPGA** |
| Argmax (+ threshold check in mode A) | Pi CPU | **FPGA** |
| Class-aware NMS | Pi CPU | **FPGA** |
| Scale boxes to camera resolution, drawing, report | Pi CPU | Pi CPU |

Sigmoid is already applied on the Hailo. The scores the FPGA receives are probabilities, so the FPGA has no sigmoid.

**Measured on the Pi 5 + Hailo-8L (2026-10-31, 537 test images):** Hailo inference 30.2 ms, CPU post-processing (numpy DFL + decode + NMS) 3.45 ms median, end to end 27.2 FPS. Test mAP50 0.799 / mAP50-95 0.537 (float reference 0.807 / 0.550).

**One-core C++ baseline (bit-exact, same math as this spec, 30 real frames):** full frame **2.910 ms** median / 2.951 ms P95, of which DFL 2.637 ms (91%), decode 0.278 ms, NMS 0.0006 ms. Mode B split: Pi prefilter 0.108 ms, CPU-only candidate pipeline 0.112 ms. Details in `cpu_baseline/CPU_BASELINE.md`.

## 2. Inputs (per frame, from Hailo via the Pi)

Two tensors of the Hailo's native quantized integers (uint8). The FPGA never needs floating point.

| Tensor | Shape | Meaning | Real value |
|---|---|---|---|
| `bbox_q` | 18,900 × 64 | 4 sides (l, t, r, b) × 16 DFL bins, logits | `0.173810 · (q − 171)` |
| `score_q` | 18,900 × 6 | class probabilities | `q / 255` (scale 0.0039216, zp 0) |

**HEF layout:** the HEF has **6 outputs**, one bbox (H×W×64) and one score (H×W×6) per stride: 120×120, 60×60, 30×30. The Pi flattens each one row-major and concatenates them in stride order, so the FPGA sees **one** `bbox_q` stream and **one** `score_q` stream, exactly as in the earlier 2-output design. The three bbox outputs were compiled with one shared quantization range, and the Pi measured identical params on all three (scale 0.17380952835083008, zp 171), so **one** scale / zero-point applies to every anchor. Same for the three score outputs (scale 1/255, zp 0).

**Anchor order:** stride 8 grid, then stride 16, then stride 32. Each grid is `960/s × 960/s`, row-major (y outer, x inner). Anchor (x, y) at stride `s` has center `((x+0.5)·s, (y+0.5)·s)`. Index ranges: stride 8 = 0..14,399, stride 16 = 14,400..17,999, stride 32 = 18,000..18,899.

**Hex file order:** `in_bbox.hex` is anchor → side (l, t, r, b) → bin, one value per line. `in_score.hex` is anchor → class.

**Link:** the Pi 5's only PCIe lane is used by the AI HAT, so the Pi → FPGA link is SPI (see `SPI_INTERFACE.md`). A full 960 frame is 1.32 MB, about 1.06 s at 10 MHz SCLK, so **only mode B is usable**: the Pi sends only the anchors whose max score passes the threshold (11 per frame on average, 42 at most, on the 30 real frames in this package).

**Memory:** a full frame (1.32 MB) is far more than the XC7A35T's 225 KB of block RAM. Stream DFL/decode anchor by anchor with no frame buffer, and buffer only the candidates.

## 3. Algorithm (bit-accurate; matches `golden_model.py::fixed_ref`)

**3.1 DFL (per anchor, per side, 16 bins `q[0..15]`): resolution-independent**
```
qmax   = max(q[i])
d[i]   = min(qmax - q[i], LUT_DEPTH-1)
e[i]   = LUT[d[i]]                       // Q0.15, LUT[0] = 32768, from dfl_exp_lut.mem
S      = Σ e[i]                          // 20 bits
N      = Σ i·e[i]                        // 23 bits
off    = floor((N << 8) / S)             // Q4.8, 0..3840, 12 bits  -> checkpoint exp_dfl.hex
```
The softmax max-subtraction cancels the zero-point, so the LUT is indexed by an integer difference. One divider per side; the four sides can share one. The LUT depends on the bbox scale, so it is regenerated whenever the HEF is recompiled.

**3.2 Box decode (pixels, Q.4 = 1/16 px)**
```
cx     = ((2x+1) << log2(s)) << 3        // (x+0.5)·s in Q.4
cy     = ((2y+1) << log2(s)) << 3
offpx  = (off << log2(s)) >> 4           // floor
x1 = clip(cx - offpx_l, 0, 960<<4)   y1 = clip(cy - offpx_t, 0, 960<<4)
x2 = clip(cx + offpx_r, 0, 960<<4)   y2 = clip(cy + offpx_b, 0, 960<<4)
```
Width before the clip is `COORD_S_BITS` = 16 (signed); after it, `COORD_BITS` = 14 (unsigned, max 15,360).

**3.3 Class + threshold**
```
cls    = argmax(score_q[0..5])               // lowest index wins ties
sq     = max(score_q)
pass   = (sq >= SCORE_Q_THR)                 // SCORE_Q_THR = 64 -> checkpoint exp_decode.hex
```
`SCORE_Q_THR = 64` is the quantized equivalent of conf = 0.25 (`q/255 > 0.25` ⇔ `q ≥ 64`; the strict `>` of the float code is already folded in). In mode B the Pi applies the same test before sending, so every anchor the FPGA receives passes.

**3.4 NMS (class-aware, greedy)**
- Candidates = passing anchors, in anchor order, capped at `MAX_CAND` = 4096 (set the overflow flag if more pass).
- For each class: sort candidates by `sq` descending, ties broken by lower anchor index. Then walk the list: keep the top box and suppress any later box of the same class where
  ```
  inter > 0  AND  IOU_DEN·inter >= IOU_NUM·union      // IoU >= 0.45 = 9/20, no divider
  inter = max(0, min(x2)-max(x1)) · max(0, min(y2)-max(y1))
  union = area_i + area_j - inter                      // AREA_BITS = 28 wide
  ```
- Output order: class ascending, then kept order. Capped at `MAX_DET` = 100. → checkpoint `exp_nms.hex`
- NMS cost grows with candidates² per class. Real frames had at most 42 candidates.

## 4. Output (per detection)

`x1, y1, x2, y2` (Q.4, 960×960 input-pixel space), `cls` (3 bits), `score_q`, `anchor_idx` (15 bits, for debugging), plus `n_det` and an `overflow` flag per frame. The Pi converts to camera pixels (÷16, then undo the letterbox: subtract the pad, divide by the letterbox scale).

## 5. Files in this package

| File | Purpose |
|---|---|
| `README.md` | Start here: what to build, what passes |
| `SPEC.md` | This document |
| `SPI_INTERFACE.md` | Pi ⇄ FPGA SPI link, packet format, wiring (updated for 960 / mode B) |
| `golden_model.py` | Float reference (same math as the deployed Pi code) + bit-accurate integer model. Generates everything in `rtl_vectors/` |
| `rtl_vectors/rtl_params.vh` | `localparam`s: geometry, bit widths, quant-derived threshold, IoU fraction, caps |
| `rtl_vectors/dfl_exp_lut.mem` | exp LUT (16-bit hex per line, 65 entries) for `$readmemh` |
| `rtl_vectors/frame_XXXX/in_*.hex` | Testbench stimulus (full frame, 18,900 anchors) |
| `rtl_vectors/frame_XXXX/exp_dfl.hex` | Checkpoint 1: `off_l off_t off_r off_b` per anchor |
| `rtl_vectors/frame_XXXX/exp_decode.hex` | Checkpoint 2: `x1 y1 x2 y2 cls score pass` per anchor |
| `rtl_vectors/frame_XXXX/exp_nms.hex` | Checkpoint 3: line 0 = `n_det overflow`, then `x1 y1 x2 y2 cls score anchor` |
| `rtl_vectors/frame_XXXX/exp_nms.txt` | Readable detections + comparison with the float reference |
| `rtl_vectors/summary.txt` | Per-frame result: **30/30 frames match**, max box error 0.120 px |
| `fpga_vectors/` | The raw input: real uint8 Hailo outputs (`frame_XXXX.npz`, already joined to 18,900 × 64 / × 6), `meta.json` (quant params), and the source image of each frame (`.png`) |
| `dump_vectors.py`, `maritime_post.py` | How `fpga_vectors/` was made on the Pi (reference only; needs the Hailo) |

30 frames, taken from the maritime6 test set. `$readmemh` reads whitespace-separated values sequentially, so multi-field lines load into a flat array (index = row·fields + field).

## 6. Generating vectors

```bash
# On the Pi (needs hailo_platform and the HEF):
python3 dump_vectors.py --hef maritime6_960.hef --images test/images --labels test/labels --frames 30 --out fpga_vectors
# Anywhere with numpy (this is how rtl_vectors/ was made). Pass --conf 0.25: the script's default is still the powerline 0.3.
python3 golden_model.py --vectors fpga_vectors --out rtl_vectors --conf 0.25 --iou 0.45
```
`summary.txt` reports per frame whether the integer model matches the float reference. On this package's 30 real frames: 30/30 match, max box error 0.120 px, no cap hit.

**Pass criterion for the RTL:** exact match with `exp_*.hex` at all three checkpoints. In mode B, compare `exp_dfl.hex` / `exp_decode.hex` only for the anchors that were sent (those with `pass = 1` in `exp_decode.hex`); `exp_nms.hex` must match in full.

## 7. Known differences from the deployed Python

- **NMS ties:** the original `argsort()[::-1]` resolves equal scores arbitrarily. Quantized scores tie often, so the spec fixes the rule (lower anchor index first). The kept box may differ from the Pi's output by one neighbouring anchor. Both are valid.
- **Caps:** the Python has no limit. The RTL needs `MAX_CAND` / `MAX_DET` (4096 / 100 here). If `summary.txt` says `CAP HIT` on new data, raise `--max-cand` / `--max-det`.
- **Floor rounding** in the DFL division and coordinate shifts: error ≤ 0.125 px.
- Comments inside `golden_model.py` still mention the powerline model and `powerline_inference.py` / `dump_vectors_pi.py`; the math is the same as `maritime_post.py`, and size, classes and quant params are read from `fpga_vectors/meta.json`.

## 8. Open decisions for the team

1. Confirm mode B (prefilter on the Pi) for the 960 model. Mode A is not usable at 1.32 MB per frame over SPI.
2. Parallelism: anchors per cycle in DFL, and how candidates are buffered/sorted for NMS
3. Whether conf/IoU should be runtime registers instead of `localparam`s (regenerate `SCORE_Q_THR` if conf changes)
4. Whether to synthesize one build per model or size the hardware for the largest model
5. Re-run `dump_vectors.py` + `golden_model.py` after any HEF recompile: the quant params change and so do the LUT and threshold
