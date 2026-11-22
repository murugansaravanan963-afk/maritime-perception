`timescale 1ns / 1ps

// ================================================================
// DE10-Lite YOLOv8 Mode-B top with an OVERSAMPLED SPI Mode-0 slave
//
// Every flip-flop in this file is clocked by clk_100.
// SPI_SCLK is NOT used as a clock. SCLK / CS_N / MOSI are
// synchronized into clk_100 and edge-detected.
//
// Why:
//   The previous version clocked logic on BOTH posedge and negedge
//   of the raw SPI_SCLK pin and passed state back and forth between
//   the two edge domains (byte_done_pending, tx_current_valid,
//   tx_read_pending ...). That failed byte-exact even in ModelSim
//   (bytes shifted / duplicated), and on hardware a GPIO pin used as
//   a clock is also exposed to ringing / double-clocking and is not
//   timing-constrained.
//
// Limits:
//   clk_100 / SCLK must stay >= ~8. Detection + MISO update takes
//   ~4 clk_100 cycles (~40 ns) after the real SCLK falling edge,
//   and must finish before the master's next rising edge.
//   1 MHz .. 4 MHz is comfortable; 8 MHz is about the ceiling.
//
// Read protocol (unchanged):
//   byte 0 of every transaction = dummy 0x00
//   byte 1.. = result bytes from the TX FIFO (0x00 if FIFO empty)
// ================================================================

module de10_lite_yolov8_top (
    input  logic clk_100,
    input  logic rst_n,

    input  logic spi_sclk,
    input  logic spi_cs_n,
    input  logic spi_mosi,
    output logic spi_miso,

    output logic result_ready,

    // Level signal for the host (wire to a Pi GPIO):
    //   goes HIGH when a complete result packet is in the TX FIFO,
    //   goes LOW when the host starts the SPI read of it (CS low).
    output logic result_pending,

    // TX-path debug outputs
    output logic dbg_core_tx_valid,
    output logic dbg_tx_fifo_write,
    output logic dbg_tx_fifo_nonempty,
    output logic dbg_tx_current_valid
);

    // ============================================================
    // Input synchronizers (clk_100 domain)
    // ============================================================

    (* altera_attribute = "-name SYNCHRONIZER_IDENTIFICATION FORCED_IF_ASYNCHRONOUS" *)
    logic [2:0] sclk_sync;
    (* altera_attribute = "-name SYNCHRONIZER_IDENTIFICATION FORCED_IF_ASYNCHRONOUS" *)
    logic [2:0] cs_sync;
    (* altera_attribute = "-name SYNCHRONIZER_IDENTIFICATION FORCED_IF_ASYNCHRONOUS" *)
    logic [1:0] mosi_sync;

    always_ff @(posedge clk_100 or negedge rst_n) begin
        if (!rst_n) begin
            sclk_sync <= 3'b000;
            cs_sync   <= 3'b111;
            mosi_sync <= 2'b00;
        end
        else begin
            sclk_sync <= {sclk_sync[1:0], spi_sclk};
            cs_sync   <= {cs_sync[1:0],   spi_cs_n};
            mosi_sync <= {mosi_sync[0],   spi_mosi};
        end
    end

    // sclk_sync[1] / mosi_sync[1] / cs_sync[1] are time-aligned.
    wire spi_active = !cs_sync[1];
    wire sclk_rise  = spi_active &&  sclk_sync[1] && !sclk_sync[2];
    wire sclk_fall  = spi_active && !sclk_sync[1] &&  sclk_sync[2];
    wire mosi_bit   = mosi_sync[1];


    // ============================================================
    // SPI RX: sample MOSI on SCLK rising edge, MSB first
    // ============================================================

    logic [7:0] rx_shift;
    logic [2:0] rx_bit_count;

    logic [7:0] spi_fifo_data;
    logic       spi_fifo_wr_en;

    always_ff @(posedge clk_100 or negedge rst_n) begin
        if (!rst_n) begin
            rx_shift       <= 8'h00;
            rx_bit_count   <= 3'd0;
            spi_fifo_data  <= 8'h00;
            spi_fifo_wr_en <= 1'b0;
        end
        else begin
            spi_fifo_wr_en <= 1'b0;

            if (!spi_active) begin
                // CS high: drop any partial byte, realign.
                rx_bit_count <= 3'd0;
            end
            else if (sclk_rise) begin
                rx_shift     <= {rx_shift[6:0], mosi_bit};
                rx_bit_count <= rx_bit_count + 3'd1;

                if (rx_bit_count == 3'd7) begin
                    spi_fifo_data  <= {rx_shift[6:0], mosi_bit};
                    spi_fifo_wr_en <= 1'b1;
                end
            end
        end
    end


    // ============================================================
    // RX FIFO (both sides clk_100 -> behaves as a sync FIFO)
    // ============================================================

    logic       rx_fifo_full;
    logic [7:0] rx_fifo_data;
    logic       rx_fifo_rd_en;
    logic       rx_fifo_empty;

    async_fifo_8bit #(
        .ADDR_WIDTH(11)
    ) u_rx_fifo (
        .wr_clk   (clk_100),
        .wr_rst_n (rst_n),

        .wr_data  (spi_fifo_data),
        .wr_en    (spi_fifo_wr_en),
        .wr_full  (rx_fifo_full),

        .rd_clk   (clk_100),
        .rd_rst_n (rst_n),

        .rd_data  (rx_fifo_data),
        .rd_en    (rx_fifo_rd_en),
        .rd_empty (rx_fifo_empty)
    );


    // ============================================================
    // RX FIFO -> Core adapter
    // ============================================================

    logic       core_rx_valid;
    logic [7:0] core_rx_data;
    logic       core_rx_ready;

    logic rx_read_pending;

    assign rx_fifo_rd_en =
        !rx_fifo_empty &&
        !rx_read_pending &&
        !core_rx_valid;

    always_ff @(posedge clk_100 or negedge rst_n) begin
        if (!rst_n) begin
            rx_read_pending <= 1'b0;
            core_rx_valid   <= 1'b0;
            core_rx_data    <= 8'h00;
        end
        else begin
            if (core_rx_valid && core_rx_ready)
                core_rx_valid <= 1'b0;

            if (rx_fifo_rd_en)
                rx_read_pending <= 1'b1;

            if (rx_read_pending) begin
                core_rx_data    <= rx_fifo_data;
                core_rx_valid   <= 1'b1;
                rx_read_pending <= 1'b0;
            end
        end
    end


    // ============================================================
    // Mode-B Core
    // ============================================================

    logic core_result_ready;
    logic core_result_read_start;

    logic       core_tx_ready;
    logic       core_tx_valid;
    logic [7:0] core_tx_data;
    logic       core_tx_last;

    assign result_ready = core_result_ready;

    yolov8_mode_b_packet_top u_core (
        .clk               (clk_100),
        .rst_n             (rst_n),

        .rx_valid          (core_rx_valid),
        .rx_ready          (core_rx_ready),
        .rx_data           (core_rx_data),

        .result_ready      (core_result_ready),
        .result_read_start (core_result_read_start),

        .tx_ready          (core_tx_ready),
        .tx_valid          (core_tx_valid),
        .tx_data           (core_tx_data),
        .tx_last           (core_tx_last)
    );


    // ============================================================
    // TX FIFO (both sides clk_100)
    // ============================================================

    logic       tx_fifo_full;
    logic [7:0] tx_fifo_data;
    logic       tx_fifo_rd_en;
    logic       tx_fifo_empty;

    async_fifo_8bit #(
        .ADDR_WIDTH(10)
    ) u_tx_fifo (
        .wr_clk   (clk_100),
        .wr_rst_n (rst_n),

        .wr_data  (core_tx_data),
        .wr_en    (core_tx_valid && core_tx_ready),
        .wr_full  (tx_fifo_full),

        .rd_clk   (clk_100),
        .rd_rst_n (rst_n),

        .rd_data  (tx_fifo_data),
        .rd_en    (tx_fifo_rd_en),
        .rd_empty (tx_fifo_empty)
    );

    assign core_tx_ready = !tx_fifo_full;


    // ============================================================
    // Start result builder exactly once
    // ============================================================

    logic result_start_sent;

    always_ff @(posedge clk_100 or negedge rst_n) begin
        if (!rst_n) begin
            core_result_read_start <= 1'b0;
            result_start_sent      <= 1'b0;
        end
        else begin
            core_result_read_start <= 1'b0;

            if (!core_result_ready)
                result_start_sent <= 1'b0;

            if (core_result_ready && !result_start_sent) begin
                core_result_read_start <= 1'b1;
                result_start_sent      <= 1'b1;
            end
        end
    end


    // ============================================================
    // TX prefetch: keep one byte ready in tx_next_byte
    //
    // The FIFO read port is registered (1-cycle latency), so a read
    // is issued, then captured on the following clk_100 cycle.
    // At 100 MHz this completes long before the SPI byte boundary.
    // ============================================================

    logic [7:0] tx_next_byte;
    logic       tx_next_valid;
    logic       tx_read_pending;
    logic       tx_take_next;      // serializer consumed tx_next_byte

    assign tx_fifo_rd_en =
        !tx_fifo_empty &&
        !tx_read_pending &&
        !tx_next_valid;

    always_ff @(posedge clk_100 or negedge rst_n) begin
        if (!rst_n) begin
            tx_next_byte    <= 8'h00;
            tx_next_valid   <= 1'b0;
            tx_read_pending <= 1'b0;
        end
        else begin
            if (tx_take_next)
                tx_next_valid <= 1'b0;

            if (tx_fifo_rd_en)
                tx_read_pending <= 1'b1;

            if (tx_read_pending) begin
                tx_next_byte    <= tx_fifo_data;
                tx_next_valid   <= 1'b1;
                tx_read_pending <= 1'b0;
            end
        end
    end


    // ============================================================
    // TX serializer, SPI Mode 0 (CPOL=0, CPHA=0), MSB first
    //
    //   MISO = tx_shift[7] at all times while CS is low.
    //   Rising edge : master samples, we count the bit.
    //   Falling edge: shift to next bit, or load the next byte
    //                 after the 8th bit has been sampled.
    //
    // Byte 0 of each transaction is the dummy 0x00. Because its
    // MSB is 0, the CS-low -> first-SCLK setup time does not matter.
    // ============================================================

    logic [7:0] tx_shift;
    logic [2:0] tx_bit_count;
    logic       tx_byte_done;   // 8th bit sampled, load on next fall
    logic       tx_dummy_phase; // still sending dummy byte

    // ------------------------------------------------------------
    // tx_serve: result bytes are only sent in a transaction that
    // STARTED with result data already available.
    //
    // The core finishes the result while CS is still low for the
    // input-packet write. Without this gate the first result
    // bytes would be shifted out (and lost) during that write.
    // ------------------------------------------------------------
    logic spi_active_d;
    logic tx_serve;

    wire tx_data_available =
        tx_next_valid || tx_read_pending || !tx_fifo_empty;

    assign tx_take_next =
        tx_serve && sclk_fall && tx_byte_done && tx_next_valid;

    always_ff @(posedge clk_100 or negedge rst_n) begin
        if (!rst_n) begin
            spi_active_d <= 1'b0;
            tx_serve     <= 1'b0;
        end
        else begin
            spi_active_d <= spi_active;

            if (!spi_active)
                tx_serve <= 1'b0;
            else if (!spi_active_d)        // CS just went low
                tx_serve <= tx_data_available;
        end
    end

    // ------------------------------------------------------------
    // result_pending (host-visible "ready to read" level)
    // ------------------------------------------------------------
    always_ff @(posedge clk_100 or negedge rst_n) begin
        if (!rst_n)
            result_pending <= 1'b0;
        else if (core_tx_valid && core_tx_ready && core_tx_last)
            result_pending <= 1'b1;        // last result byte queued
        else if (spi_active && !spi_active_d && tx_data_available)
            result_pending <= 1'b0;        // host started reading it
    end

    always_ff @(posedge clk_100 or negedge rst_n) begin
        if (!rst_n) begin
            tx_shift       <= 8'h00;
            tx_bit_count   <= 3'd0;
            tx_byte_done   <= 1'b0;
            tx_dummy_phase <= 1'b1;
        end
        else if (!spi_active) begin
            // Idle / between transactions: arm the dummy byte.
            tx_shift       <= 8'h00;
            tx_bit_count   <= 3'd0;
            tx_byte_done   <= 1'b0;
            tx_dummy_phase <= 1'b1;
        end
        else begin
            if (sclk_rise) begin
                tx_bit_count <= tx_bit_count + 3'd1;
                if (tx_bit_count == 3'd7)
                    tx_byte_done <= 1'b1;
            end

            if (sclk_fall) begin
                if (tx_byte_done) begin
                    tx_byte_done   <= 1'b0;
                    tx_dummy_phase <= 1'b0;
                    tx_shift       <= tx_take_next ? tx_next_byte : 8'h00;
                end
                else begin
                    tx_shift <= {tx_shift[6:0], 1'b0};
                end
            end
        end
    end

    // Registered MISO bit, gated by the raw CS so it is 0 when idle.
    assign spi_miso = !spi_cs_n ? tx_shift[7] : 1'b0;


    // ============================================================
    // TX debug outputs (all clk_100 domain)
    // ============================================================

    assign dbg_core_tx_valid    = core_tx_valid;
    assign dbg_tx_fifo_write    = core_tx_valid && core_tx_ready;
    assign dbg_tx_fifo_nonempty = !tx_fifo_empty;
    assign dbg_tx_current_valid = tx_take_next;


`ifndef SYNTHESIS

    always_ff @(posedge clk_100) begin
        if (rst_n && spi_fifo_wr_en && rx_fifo_full)
            $error("de10_lite_yolov8_top: RX FIFO overflow");

        if (rst_n && core_tx_valid && tx_fifo_full)
            $error("de10_lite_yolov8_top: TX FIFO overflow");
    end

`ifdef SPI_DEBUG
    always_ff @(posedge clk_100) begin
        if (spi_fifo_wr_en)
            $display("[SPI RXB] 0x%02h  t=%0t", spi_fifo_data, $time);
        if (core_rx_valid && core_rx_ready)
            $display("[CORE RX] 0x%02h  t=%0t", core_rx_data, $time);
        if (tx_take_next)
            $display("[SPI TXB] 0x%02h  t=%0t", tx_next_byte, $time);
    end
`endif

`endif

endmodule
