# ============================================================
# Virtual FPGA Mode-B - ModelSim Batch Runner
#
# Expected directory structure:
#
# project_root/
#   mode_b_v1/
#   RTL_v6/
#   virtual_fpga_mode_b/
#       run_mode_b.do
#       mode_b_socket_tb.sv
#
# ============================================================

onerror {
    puts "========================================================"
    puts " MODE-B MODELSIM RUN FAILED"
    puts "========================================================"
    quit -code 1 -f
}

# ------------------------------------------------------------
# 1. Locate directories relative to this .do file
# ------------------------------------------------------------

# ModelSim is launched by mode_b_bridge.py with
# cwd = virtual_fpga_mode_b
set SCRIPT_DIR [file normalize [pwd]]
set ROOT_DIR   [file normalize [file join $SCRIPT_DIR ".."]]

set R [file normalize [file join $ROOT_DIR "RTL_v6"]]
set M [file normalize [file join $ROOT_DIR "mode_b_v1"]]
set T $SCRIPT_DIR

puts ""
puts "========================================================"
puts " VIRTUAL FPGA MODE-B MODELSIM RUN"
puts "========================================================"
puts " ROOT_DIR   = $ROOT_DIR"
puts " RTL_DIR    = $R"
puts " MODE_B_DIR = $M"
puts " TB_DIR     = $T"
puts "========================================================"

# ------------------------------------------------------------
# 2. Check important files/directories
# ------------------------------------------------------------

if {![file isdirectory $R]} {
    puts "ERROR: RTL_v6 directory not found:"
    puts "       $R"
    quit -code 1 -f
}

if {![file isdirectory $M]} {
    puts "ERROR: mode_b_v1 directory not found:"
    puts "       $M"
    quit -code 1 -f
}

if {![file exists [file join $R "rtl_params.vh"]]} {
    puts "ERROR: rtl_params.vh not found."
    quit -code 1 -f
}

if {![file exists [file join $R "dfl_exp_lut.mem"]]} {
    puts "ERROR: dfl_exp_lut.mem not found."
    quit -code 1 -f
}

if {![file exists [file join $T "mode_b_socket_tb.sv"]]} {
    puts "ERROR: mode_b_socket_tb.sv not found."
    quit -code 1 -f
}

puts "\[MODELSIM\] Source directories OK."

# ------------------------------------------------------------
# 3. Create clean dedicated library
# ------------------------------------------------------------

set LIB_DIR [file normalize [file join $T "work_virtual"]]

if {[file exists $LIB_DIR]} {
    puts "\[MODELSIM\] Removing old work_virtual..."
    file delete -force $LIB_DIR
}

puts "\[MODELSIM\] Creating work_virtual..."

vlib $LIB_DIR
vmap work $LIB_DIR

puts "\[MODELSIM\] Library ready."

# ------------------------------------------------------------
# 4. Compile Frozen V6 compute RTL
# ------------------------------------------------------------

puts ""
puts "========================================================"
puts " COMPILING FROZEN V6 COMPUTE RTL"
puts "========================================================"

vlog -work work -sv +incdir+$R \
    [file join $R "downscale_block.v"] \
    [file join $R "dfl_exp_lut.v"] \
    [file join $R "exp_adder_tree.v"] \
    [file join $R "dfl_divider.v"] \
    [file join $R "dfl_core.sv"] \
    [file join $R "yolov8_dfl_top.sv"] \
    [file join $R "bbox_decode.sv"] \
    [file join $R "score_argmax_threshold.sv"] \
    [file join $R "yolov8_decode_top.sv"] \
    [file join $R "candidate_collector.sv"] \
    [file join $R "iou_compare.sv"] \
    [file join $R "nms_engine.sv"] \
    [file join $R "yolov8_nms_top.sv"]

puts "\[MODELSIM\] Frozen V6 compute RTL compiled."

# ------------------------------------------------------------
# 5. Compile Mode-B RTL
# ------------------------------------------------------------

puts ""
puts "========================================================"
puts " COMPILING MODE-B RTL"
puts "========================================================"

vlog -work work -sv +incdir+$R +incdir+$M \
    [file join $M "mode_b_crc16_ccitt.sv"] \
    [file join $M "mode_b_packet_parser.sv"] \
    [file join $M "mode_b_sparse_decode.sv"] \
    [file join $M "mode_b_compute_top.sv"] \
    [file join $M "mode_b_result_builder.sv"] \
    [file join $M "yolov8_mode_b_packet_top.sv"]

puts "\[MODELSIM\] Mode-B RTL compiled."

# ------------------------------------------------------------
# 6. Compile virtual FPGA socket/file TB
# ------------------------------------------------------------

puts ""
puts "========================================================"
puts " COMPILING VIRTUAL FPGA TESTBENCH"
puts "========================================================"

vlog -work work -sv \
    [file join $T "mode_b_socket_tb.sv"]

puts "\[MODELSIM\] Testbench compiled."

# ------------------------------------------------------------
# 7. Change working directory
#
# Python bridge writes:
#   virtual_fpga_mode_b/rx_packet.bin
#
# RTL writes:
#   virtual_fpga_mode_b/tx_reply.bin
# ------------------------------------------------------------

cd $T

puts ""
puts "\[MODELSIM\] Simulation working directory:"
puts "           [pwd]"

# ------------------------------------------------------------
# 8. Start simulation
# ------------------------------------------------------------

puts ""
puts "========================================================"
puts " STARTING MODE-B VIRTUAL FPGA SIMULATION"
puts "========================================================"

vsim -c work.mode_b_socket_tb

run -all

puts ""
puts "========================================================"
puts " MODE-B VIRTUAL FPGA SIMULATION FINISHED"
puts "========================================================"

quit -f