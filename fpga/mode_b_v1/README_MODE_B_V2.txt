MODE-B V2 - Empty-frame support
================================
Changes from V1:
1) mode_b_compute_top.sv
   - Adds an empty-frame frame_end pulse when frame_start && n_candidates==0.
   - Frozen yolov8_dfl_top and yolov8_nms_top are unchanged.
2) mode_b_packet_parser.sv
   - For payload_len==0, last_payload_pulse is asserted with frame_start so cycle accounting has a defined payload boundary.
3) yolov8_mode_b_empty_frame_tb.sv
   - Sends a valid 18-byte empty Mode-B frame packet (16-byte header + CRC).
   - Expected reply: 14 bytes, status=0x01, same frame_id, n_det=0, valid CRC.

Regression order:
A. Run yolov8_mode_b_empty_frame_tb -> PASS.
B. Run existing yolov8_mode_b_frame0000_tb with the V2 parser/compute top -> must remain PASS (18 candidates, 1 detection).
C. Re-run Pi/Hailo 30-frame virtual-FPGA test -> target 30/30 PASS, including the two empty frames.
