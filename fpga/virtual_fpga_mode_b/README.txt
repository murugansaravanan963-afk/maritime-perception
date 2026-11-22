VIRTUAL FPGA MODE-B LOCALHOST TEST
=================================
Purpose: functional TCP -> ModelSim RTL -> TCP validation. It does NOT measure real FPGA/SPI latency.

Required existing folders:
  1) Final RTL_v6 folder (contains rtl_params.vh and dfl_exp_lut.mem)
  2) mode_b_v2 folder (contains the Mode-B .sv files that already passed frame0000)

TCP framing:
  4-byte big-endian TCP length + unchanged Y8PP Mode-B packet.
The 4-byte length exists only for TCP. It is NOT part of the SPI protocol.

Windows terminal 1 (example):
  python mode_b_bridge.py --rtl-dir "C:\path\RTL_v6" --mode-b-dir "C:\path\mode_b_v2"

Windows terminal 2:
  python mode_b_local_client.py --frame-dir "C:\path\RTL_v6\frame_0000"

Expected first test:
  client candidates=18
  bridge launches ModelSim batch
  reply=26 bytes, status=0x01, n_det=1
  one detection matching the Mode-B frame0000 result

If 'vsim' is not on PATH, pass e.g.:
  --vsim "C:\intelFPGA\...\win64\vsim.exe"
(or use the actual ModelSim/Questa executable path installed on your PC).

Files generated under ./io:
  rx_packet.hex
  tx_reply.hex
ModelSim transcript:
  ./io/modelsim_bridge.log (depending on ModelSim transcript behavior)

Later Pi use:
  Replace mode_b_local_client.py with the Pi C++ sender. Keep the Y8PP packet unchanged.
  For remote Pi/PC, bind bridge to --host 0.0.0.0 and use a reachable PC IP/VPN address.
