# Pi 5 ⇄ XC7A35T SPI interface (proposal v2, 2026-10-31: Maritime6 960)

**v2 changes:** target model is now Maritime6 960 (18,900 anchors, 6 classes). A full frame is
1.32 MB, so **mode B (Pi prefilters, sends only passing anchors) is the required mode**; mode A
stays in the spec for bring-up only. Transfer numbers below are recomputed for 960. Powerline
(416) numbers from v1 are kept in brackets where useful.

Scope: the DFL + decode + threshold + NMS block from `SPEC.md`, fed over SPI by the
Raspberry Pi 5. The Hailo-8L keeps the PCIe lane; the FPGA hangs off the 40-pin header.
Nothing here is measured on hardware yet; numbers marked *calc* are arithmetic.

## 1. Answers

| # | Question | Decision |
|---|---|---|
| 1 | SPI clock | Bring-up **1 MHz**, then **10 MHz** target, try **20 MHz** only with short wires and a clean scope trace. The Pi driver rounds the requested speed down to what RP1 can generate: confirm the real SCLK on a scope/logic analyzer and report that value. |
| 2 | Mode | **Mode 0** (CPOL=0, CPHA=0), 8-bit words, MSB first, CE0 active low. |
| 3 | Stream or frame | **One frame per transaction, streamed anchor by anchor, mode B (only passing anchors).** The FPGA never stores the full frame (1.32 MB at 960 [298 KB at 416] > the XC7A35T's 225 KB of BRAM). |
| 4 | Packet format | Section 3 below. |
| 5 | READY/IRQ GPIO | **Yes.** `DONE` (FPGA→Pi) is required; `BUSY` and `RESET_N` are recommended. Section 4. |
| 6 | Pi program | **C++17, `/dev/spidev0.0` via `ioctl(SPI_IOC_MESSAGE)`, GPIO via libgpiod v2.** Not Python, and not RPi.GPIO (it does not support the Pi 5's RP1 GPIO). |

## 2. The transfer cost (read this before choosing the design)

**Mode A (full frame).** One 960×960 frame = 18,900 anchors × (64 bbox + 6 score) bytes =
**1,323,000 B** (*calc*) [416 powerline: 298,116 B].

| SCLK | Time to send one full frame (*calc*, payload only) | Max frame rate |
|---|---|---|
| 1 MHz | 10,584 ms | 0.09 FPS |
| 10 MHz | 1,058 ms | 0.9 FPS |
| 20 MHz | 529 ms | 1.9 FPS |
| 25 MHz | 423 ms | 2.4 FPS |

Mode A is not usable at 960. Keep it only for bring-up with the full-frame testbench vectors.

**Mode B (required).** The Pi sends only anchors whose max score ≥ `SCORE_Q_THR` (= 64,
conf 0.25), each as `anchor_idx(2) + bbox(64) + score(6)` = **72 B**. On the 30 real
maritime frames in `rtl_vectors/` the candidate count was **11.1 on average, 42 at most**:

| Candidates | Packet (16 header + 72·n + 2 CRC) | 10 MHz | 20 MHz |
|---|---|---|---|
| 0 | 18 B | 0.01 ms | 0.01 ms |
| 11 (mean) | 810 B | 0.65 ms | 0.32 ms |
| 42 (max seen) | 3,042 B | 2.43 ms | 1.22 ms |
| 100 | 7,218 B | 5.77 ms | 2.89 ms |

Busy scenes give more candidates, so measure on the demo videos too. Mode B moves
thresholding to the Pi, so the FPGA scope becomes DFL + decode + NMS, and the checkpoints
`exp_dfl.hex` / `exp_decode.hex` are then only compared for the sent anchors (`pass = 1`).
The Pi's prefilter (max over 6 scores × 18,900 anchors + gather) must be timed and counted
on the Pi side of the comparison.

**CPU baseline (one Pi 5 core, measured 2026-10-31, `cpu_baseline/CPU_BASELINE.md`).**

| | Median | P95 |
|---|---|---|
| C++ full frame (DFL on all 18,900 anchors + decode + NMS; same work as mode A) | 2.910 ms | 2.951 ms |
| C++ prefilter (argmax + threshold; the Pi side of mode B) | 0.108 ms | 0.109 ms |
| C++ prefilter + pack SPI buffer | 0.109 ms | 0.110 ms |
| C++ candidate pipeline (prefilter + DFL/decode/NMS on candidates only, CPU-only) | 0.112 ms | 0.115 ms |
| Python/numpy deployed code (`pi_baseline.py`) | 3.45 ms | |

So in mode B the Pi pays 0.108 ms for the prefilter plus the SPI time (≈ 0.65 ms at 10 MHz for
an average frame) before the FPGA computes anything, while the CPU alone finishes in 0.112 ms.
Over SPI the FPGA path cannot win end to end; that is a property of the link. Report link and
compute separately:

- **FPGA compute**: cycles from last payload byte to `DONE`, and from first byte to `DONE`.
- **Link**: measured SPI time per frame on the Pi (plus the Pi prefilter time in mode B).

## 3. Packet format

All multi-byte fields are **big-endian** (matches MSB-first shifting). One SPI transaction
(CE0 low → high) may carry any number of bytes; the FPGA parser keeps state across CE0
toggles, because spidev splits large writes into ≤ `bufsiz` chunks (default 4096 B; raise
with `spidev.bufsiz=65536` in `/boot/firmware/cmdline.txt`). Framing comes from the header,
not from CE0.

### 3.1 Pi → FPGA: FRAME (cmd 0x01)

| Offset | Size | Field |
|---|---|---|
| 0 | 4 | magic `0x59 0x38 0x50 0x50` ("Y8PP") |
| 4 | 1 | version = 1 |
| 5 | 1 | cmd = 0x01 FRAME (0x03 = FRAME_CANDIDATES, mode B) |
| 6 | 2 | frame_id (wraps) |
| 8 | 2 | n_anchors in this packet (18,900 for mode A; candidate count for mode B) |
| 10 | 1 | n_classes (6 maritime; 20 powerline) |
| 11 | 1 | bins = 16 |
| 12 | 4 | payload_len in bytes |
| 16 | payload_len | anchors in `SPEC.md` order; per anchor: `bbox[l,t,r,b][16]` then `score[n_classes]` (mode B: `anchor_idx u16` first) |
| 16+len | 2 | CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF) over bytes 0..15+len |

**Open: mode B record format.** This table gives 72 B per candidate (`anchor_idx u16, bbox[64],
score[6]`). `bench/prefilter_bench.cpp` packs 68 B (`anchor_idx u16, cls u8, score_q u8, bbox[64]`):
the Pi already did the argmax, so only the class and top score are sent. Timing differs by
< 0.01 ms; the team should pick one and keep the RTL, the Pi sender and the bench in sync.

Per-anchor interleaving lets the FPGA finish each anchor as it arrives. The Hailo returns
bbox and score as two separate tensors, so the Pi interleaves them; time that copy on the Pi
and count it in the link cost.

The FPGA rejects a bad magic/version/length/CRC, sets the error bit and raises `DONE`.

### 3.2 Pi → FPGA: READ_RESULT (cmd 0x02) and FPGA → Pi reply

The Pi sends `0x59 0x38 0x50 0x50 0x01 0x02` and then clocks dummy `0x00` bytes. The FPGA
starts driving MISO on the 7th byte:

| Offset | Size | Field |
|---|---|---|
| 0 | 1 | status: bit0 ok, bit1 crc_error, bit2 format_error, bit3 overflow (MAX_CAND hit), bit4 det_cap (MAX_DET hit) |
| 1 | 2 | frame_id echoed |
| 3 | 1 | n_det (≤ MAX_DET = 100) |
| 4 | 4 | cycles_first_byte_to_done |
| 8 | 4 | cycles_last_byte_to_done |
| 12 | 12 × n_det | per detection: `x1 y1 x2 y2` u16 each (Q.4 input pixels), `cls` u8, `score_q` u8, `anchor_idx` u16 |
| … | 2 | CRC-16/CCITT-FALSE over the reply |

Max reply = 12 + 1200 + 2 = 1,214 B. Read `12 + 12·n_det + 2` bytes; the Pi can read the
fixed 12-byte head first. Field widths fit the 960 model (coords ≤ 15,360 in Q.4, anchors ≤ 18,899, cls ≤ 5) with room up to 1280.
Pass criterion is unchanged: detections must equal `exp_nms.hex` exactly.

## 4. Wiring (3.3 V only, common ground)

| Signal | Pi 5 header | Dir | Notes |
|---|---|---|---|
| SCLK | GPIO11, pin 23 | Pi→FPGA | |
| MOSI | GPIO10, pin 19 | Pi→FPGA | |
| MISO | GPIO9, pin 21 | FPGA→Pi | |
| CE0 | GPIO8, pin 24 | Pi→FPGA | active low |
| DONE | GPIO25, pin 22 | FPGA→Pi | high when result ready; stays high until READ_RESULT completes |
| BUSY | GPIO24, pin 18 | FPGA→Pi | optional; high from header accept to DONE |
| RESET_N | GPIO23, pin 16 | Pi→FPGA | optional; low ≥ 1 µs resets parser state |
| GND | pin 20/25 | | at least two ground wires next to the signals |

Enable SPI with `dtparam=spi=on` in `/boot/firmware/config.txt`. The AI HAT+ uses the
header's stacking connector; confirm these GPIOs are free on your stack. Set the FPGA pins
to LVCMOS33 and check the bank's VCCO is 3.3 V on your board.

## 5. FPGA-side notes

- Planned system clock: **150 MHz** (from an MMCM; some XC7A35T boards have a 12 MHz oscillator).
- Treat SCLK, MOSI and CE0 as asynchronous: two-flop synchronizers into the system clock and
  edge detection. That needs the system clock ≥ ~4–5× SCLK, so **150 MHz allows ~30 MHz SCLK**.
  The wires and the Pi side, not the FPGA, will likely set the real limit; 20 MHz stays the
  practical target.
- 150 MHz leaves 6.67 ns per stage on an Artix-7: pipeline the DFL divider (`(N<<8)/S`, 31-bit
  by 20-bit) and the NMS IoU products (`IOU_DEN·inter`, `IOU_NUM·union`) and confirm WNS ≥ 0
  post-route. If timing fails, report the clock that closes rather than the 150 MHz target.
- Report all cycle counters together with the clock frequency the bitstream actually met.
- Mode B: 72 bytes per anchor = 576 SCLK periods per anchor (mode A: 70 B = 560); the DFL pipeline only has to keep up with that.
- A free-running cycle counter gives the two cycle fields; report them with the clock frequency.

## 6. What to record for the CPU vs FPGA comparison

| Metric | CPU (Pi 5) | FPGA (XC7A35T) |
|---|---|---|
| Compute latency | median, P95 from `pi_efficiency.py` | `cycles_last_byte_to_done / f_clk`, and first-byte |
| Link latency | n/a | measured SPI time per frame on the Pi |
| Resources | 1 core, CPU time per frame, core duty at 30 FPS | LUT, FF, BRAM, DSP from `report_utilization` (post-route) |
| Clock | measured core frequency | `f_clk`, timing met (WNS ≥ 0) |
| Power | Pi rail sum idle vs load; USB-C meter if available | `report_power` estimate + measured board power if possible |
| Energy per frame | P × t (total and above-idle) | P × compute time; separately P × link time |

Energy per frame is only comparable when both sides use the same boundary: uint8 tensors in,
detections out.
