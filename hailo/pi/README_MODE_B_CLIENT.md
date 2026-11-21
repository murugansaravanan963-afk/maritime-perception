# Pi Mode B client: real Hailo frames → virtual FPGA

```
Pi 5 + Hailo-8L ── prefilter (argmax, score ≥ 64) ── Y8PP packet ──TCP──► PC bridge ─► ModelSim Mode B RTL
        ▲                                                                                   │
        └──── check CRC, status, frame_id, and every box bit-exact vs golden_model ◄──TCP── reply
```

| File | Purpose |
|---|---|
| `pi_mode_b_client.py` | runs the HEF (or replays saved frames), sends each frame, checks the reply, writes `mode_b_results/results.json` + overlay images |
| `mode_b_protocol.py` | Y8PP packet / reply / CRC16 / TCP framing (same bytes as the RTL team's `mode_b_local_client.py`) |
| `mock_virtual_fpga.py` | fake FPGA answering with the golden model; for testing without ModelSim |
| `golden_model.py`, `maritime_post.py` | bit-exact reference and letterbox (unchanged copies) |

Verified on the laptop with the 30 real frames from `fpga_vectors`:
- frame 0 packet is byte-identical to the RTL team's `rx_packet.bin`;
- the golden reply equals their RTL reply (`tx_reply.hex`), apart from the cycle counters;
- golden detections equal the RTL's `exp_nms.txt` on 30/30 frames;
- client ↔ mock over TCP: 28/28 sent frames bit-exact (2 frames have 0 candidates, see below);
- corrupted packets, corrupted replies and wrong boxes are all detected.

## 1. Self-test on the Pi (no network, no ModelSim)

```bash
cd ~/maritime6_pi/mode_b && python3 mock_virtual_fpga.py
```

In a second terminal:

```bash
cd ~/maritime6_pi/mode_b && python3 pi_mode_b_client.py --hef ../maritime6_960.hef --images ../test/images --frames 30
```

Expect `30/30 sent frames bit-exact` (minus frames with 0 candidates). This proves the Hailo → packet path on the Pi.
Adjust the HEF and image paths to where they are on your Pi.

## 2. Connect the Pi to the RTL teammate's PC

If both are not on the same network, install **Tailscale** on both (free) and sign in with the same account
(or share the PC to your account). The PC gets an address like `100.x.y.z`.

Teammate starts the bridge so it accepts outside connections, and allows TCP port 9000 in Windows Firewall:

```
python mode_b_bridge.py --host 0.0.0.0 --port 9000
```

## 3. Real Pi → virtual FPGA

Start with 3 frames (ModelSim recompiles for every packet, so each frame takes several seconds or more):

```bash
python3 pi_mode_b_client.py --hef ../maritime6_960.hef --images ../test/images --frames 3 --host 100.x.y.z
```

Then all 30. Send `mode_b_results/results.json` to the RTL team; the overlay `.jpg` files are the demo pictures.

## 4. Real FPGA board (DE10-Lite) over SPI

Wiring (both boards off): Pi pin 23→JP1 1, 24→2, 19→3, 21←4, 25 (GND)→12. Never JP1 pin 11 (5 V).
One-time on the Pi: `sudo raspi-config` → Interface Options → SPI → Yes, reboot, `sudo apt install python3-spidev`.

a) Fixed regression packet (teammate's test, no Hailo):

```bash
cd ~/maritime6_pi/mode_b && python3 fpga_spi_test.py --speed 1000000
```

Expect `HARDWARE BYTE-EXACT REGRESSION PASSED`; repeat at 10, 25, 50 MHz.

b) Real Hailo frames through the board:

```bash
python3 pi_mode_b_client.py --hef ../maritime6_960.hef --images ../test/images --frames 3 --spi 0.0 --spi-hz 1000000
```

Then `--frames 30`, then raise `--spi-hz`. Add `--send-empty` once the board answers 0-candidate frames.
The client writes the packet in one transfer, waits `--spi-wait-us` (default 2000), then reads one dummy byte
plus the longest possible reply (1215 B) and keeps the part given by n_det. Here `round_trip_ms` is real.

## Reading results.json

| Field | Meaning |
|---|---|
| `result` | PASS = CRC, status, frame_id and every detection identical to the golden model |
| `fpga_us` | RTL cycles (last request byte → done) ÷ 150 MHz: the real FPGA compute time |
| `spi_ms_10MHz`, `spi_ms_20MHz` | request + reply bytes over SPI at that clock: estimate for the future link |
| `pi_prefilter_pack_ms` | numpy prefilter + packing on the Pi (the C++ version measured 0.11 ms) |
| `round_trip_ms_NOT_FPGA_LATENCY` | network + ModelSim wall time. Never report this as FPGA latency. |

## Open items for the RTL team

1. **Zero-candidate frames are real:** 2 of the 30 real frames (frame_0008, frame_0019) have no anchor with score ≥ 64.
   The Mode B V1 README says zero-candidate handling is not done yet, so the client does not send these frames
   (`--send-empty` sends them anyway). The RTL must answer them (status OK, n_det 0) before hardware use.
2. **Speed grade:** timing was closed on XC7A100T **-2**; Arty A7-100T / Nexys A7-100T are **-1**
   (`xc7a100tcsg324-1`). Re-run implementation for the board you will buy.
3. **SPI_INTERFACE.md** still sizes packets with 72 B per candidate; the agreed record is 68 B
   (packet = 16 + 68·n + 2 bytes).
