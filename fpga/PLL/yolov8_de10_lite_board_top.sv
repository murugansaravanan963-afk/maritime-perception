`timescale 1ns / 1ps

module yolov8_de10_lite_board_top (

    // DE10-Lite 50 MHz oscillator
    input  logic CLOCK_50,

    // KEY0, active low
    input  logic RESET_N,

    // Raspberry Pi SPI
    input  logic SPI_SCLK,
    input  logic SPI_CS_N,
    input  logic SPI_MOSI,
    output logic SPI_MISO,

    // Result-ready handshake to the Raspberry Pi (3.3 V GPIO input).
    // HIGH = a complete result is waiting; drops when the Pi starts
    // the SPI read. Assign to a free JP1 GPIO pin in Pin Planner.
    output logic RESULT_READY,

    // TX debug LEDs
    output logic [3:0] LEDR
);


    // ============================================================
    // PLL
    // ============================================================

    wire clk_100;
    wire pll_locked;
    wire core_rst_n;


    assign core_rst_n =
        RESET_N &&
        pll_locked;


    yolov8_pll u_pll (
        .inclk0 (CLOCK_50),
        .areset (!RESET_N),
        .c0     (clk_100),
        .locked (pll_locked)
    );


    // ============================================================
    // YOLOv8 signals
    // ============================================================

    logic result_ready;

    logic dbg_core_tx_valid;
    logic dbg_tx_fifo_write;
    logic dbg_tx_fifo_nonempty;
    logic dbg_tx_current_valid;


    // ============================================================
    // YOLOv8 Mode-B accelerator
    // ============================================================

    de10_lite_yolov8_top u_yolov8 (
        .clk_100      (clk_100),
        .rst_n        (core_rst_n),

        .spi_sclk     (SPI_SCLK),
        .spi_cs_n     (SPI_CS_N),
        .spi_mosi     (SPI_MOSI),
        .spi_miso     (SPI_MISO),

        .result_ready (result_ready),
        .result_pending (RESULT_READY),

        .dbg_core_tx_valid (
            dbg_core_tx_valid
        ),

        .dbg_tx_fifo_write (
            dbg_tx_fifo_write
        ),

        .dbg_tx_fifo_nonempty (
            dbg_tx_fifo_nonempty
        ),

        .dbg_tx_current_valid (
            dbg_tx_current_valid
        )
    );


    // ============================================================
    // Sticky debug LEDs
    //
    // LEDR0 = Result Builder produced TX data
    // LEDR1 = Result byte written into TX FIFO
    // LEDR2 = TX FIFO became non-empty
    // LEDR3 = SPI serializer obtained a valid result byte
    //
    // KEY0 clears all sticky flags.
    // ============================================================

    logic led_core_tx_valid;
    logic led_tx_fifo_write;
    logic led_tx_fifo_nonempty;
    logic led_tx_current_valid;


    // ============================================================
    // LEDR0
    // ============================================================

    always_ff @(posedge clk_100 or negedge core_rst_n) begin

        if (!core_rst_n)
            led_core_tx_valid <= 1'b0;

        else if (dbg_core_tx_valid)
            led_core_tx_valid <= 1'b1;

    end


    // ============================================================
    // LEDR1
    // ============================================================

    always_ff @(posedge clk_100 or negedge core_rst_n) begin

        if (!core_rst_n)
            led_tx_fifo_write <= 1'b0;

        else if (dbg_tx_fifo_write)
            led_tx_fifo_write <= 1'b1;

    end


    // ============================================================
    // LEDR2
    // ============================================================

    always_ff @(posedge clk_100 or negedge core_rst_n) begin

        if (!core_rst_n)
            led_tx_fifo_nonempty <= 1'b0;

        else if (dbg_tx_fifo_nonempty)
            led_tx_fifo_nonempty <= 1'b1;

    end


    // ============================================================
    // LEDR3
    // ============================================================

    always_ff @(posedge clk_100 or negedge core_rst_n) begin

        if (!core_rst_n)
            led_tx_current_valid <= 1'b0;

        else if (dbg_tx_current_valid)
            led_tx_current_valid <= 1'b1;

    end


    // ============================================================
    // LED outputs
    // ============================================================

    assign LEDR[0] = led_core_tx_valid;
    assign LEDR[1] = led_tx_fifo_write;
    assign LEDR[2] = led_tx_fifo_nonempty;
    assign LEDR[3] = led_tx_current_valid;


endmodule