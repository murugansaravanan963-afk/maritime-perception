# ============================================================
# DE10-Lite YOLOv8 Mode-B Board Timing Constraints
#
# Board clock:
#     CLOCK_50 = 50 MHz
#
# PLL:
#     50 MHz -> 100 MHz core clock (clk_100)
#
# SPI (oversampled slave, 2026-11-11):
#     SPI_SCLK is NOT a clock any more. SCLK / CS_N / MOSI are
#     asynchronous inputs that are synchronized into clk_100 by
#     2-3 flip-flop synchronizers and edge-detected.
#     Supported Raspberry Pi SPI clock: up to ~8 MHz.
#
# Previous dual-edge version: Yolov8_dual_edge_backup.sdc
# ============================================================


# ============================================================
# 1. DE10-Lite onboard clock: 50 MHz, period 20 ns
# ============================================================

create_clock -name CLOCK_50 \
    -period 20.000 \
    [get_ports {CLOCK_50}]


# ============================================================
# 2. PLL-generated 100 MHz core clock
# ============================================================

derive_pll_clocks

derive_clock_uncertainty


# ============================================================
# 3. Asynchronous SPI / handshake I/O
#
# Inputs go straight into synchronizers, so there is no
# meaningful setup/hold relationship to clk_100.
#
# Outputs (MISO, RESULT_READY) are registered in clk_100 and
# sampled by the Raspberry Pi half an SCLK period later
# (>= 62 ns at 8 MHz), which is far longer than any I/O delay.
# ============================================================

set_false_path -from [get_ports {SPI_SCLK SPI_CS_N SPI_MOSI}]

set_false_path -to   [get_ports {SPI_MISO RESULT_READY}]


# ============================================================
# 4. Push-button reset and debug LEDs (static / human speed)
# ============================================================

set_false_path -from [get_ports {RESET_N}]

set_false_path -to   [get_ports {LEDR[*]}]
