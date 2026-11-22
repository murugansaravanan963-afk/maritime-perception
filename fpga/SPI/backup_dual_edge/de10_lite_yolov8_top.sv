`timescale 1ns / 1ps

module de10_lite_yolov8_top (
    input  logic clk_100,
    input  logic rst_n,

    input  logic spi_sclk,
    input  logic spi_cs_n,
    input  logic spi_mosi,
    output logic spi_miso,

    output logic result_ready,

    // TX-path debug outputs
    output logic dbg_core_tx_valid,
    output logic dbg_tx_fifo_write,
    output logic dbg_tx_fifo_nonempty,
    output logic dbg_tx_current_valid
);

    // ============================================================
    // SPI RX
    // ============================================================

    logic [7:0] spi_rx_byte;
    logic       spi_rx_valid;

    logic [7:0] spi_fifo_data;
    logic       spi_fifo_wr_en;

    logic unused_spi_miso;
    logic unused_tx_done;


    spi_slave_mode0 u_spi_rx (
        .spi_sclk      (spi_sclk),
        .spi_cs_n      (spi_cs_n),
        .spi_mosi      (spi_mosi),
        .spi_miso      (unused_spi_miso),

        .rst_n         (rst_n),

        .rx_byte       (spi_rx_byte),
        .rx_byte_valid (spi_rx_valid),

        .rx_fifo_data  (spi_fifo_data),
        .rx_fifo_wr_en (spi_fifo_wr_en),

        .tx_byte       (8'h00),
        .tx_byte_load  (1'b0),

        .tx_byte_done  (unused_tx_done)
    );


    // ============================================================
    // RX ASYNC FIFO
    // SPI domain -> 100 MHz core domain
    // ============================================================

    logic       rx_fifo_full;
    logic [7:0] rx_fifo_data;
    logic       rx_fifo_rd_en;
    logic       rx_fifo_empty;


    async_fifo_8bit #(
        .ADDR_WIDTH(11)
    ) u_rx_fifo (
        .wr_clk   (spi_sclk),
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

            core_rx_valid <= 1'b0;
            core_rx_data  <= 8'h00;

        end
        else begin

            if (
                core_rx_valid &&
                core_rx_ready
            ) begin
                core_rx_valid <= 1'b0;
            end


            if (rx_fifo_rd_en) begin
                rx_read_pending <= 1'b1;
            end


            if (rx_read_pending) begin

                core_rx_data <= rx_fifo_data;
                core_rx_valid <= 1'b1;
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


    assign result_ready =
        core_result_ready;


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
    // TX ASYNC FIFO
    // 100 MHz core domain -> SPI domain
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

        .wr_en    (
            core_tx_valid &&
            core_tx_ready
        ),

        .wr_full  (tx_fifo_full),

        .rd_clk   (spi_sclk),
        .rd_rst_n (rst_n),

        .rd_data  (tx_fifo_data),
        .rd_en    (tx_fifo_rd_en),
        .rd_empty (tx_fifo_empty)
    );


    assign core_tx_ready =
        !tx_fifo_full;


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


            if (!core_result_ready) begin
                result_start_sent <= 1'b0;
            end


            if (
                core_result_ready &&
                !result_start_sent
            ) begin

                core_result_read_start <= 1'b1;
                result_start_sent      <= 1'b1;

            end

        end

    end


    // ============================================================
    // SPI TX
    //
    // Verified architecture:
    //
    //   FIFO read / byte-buffer management -> posedge SPI_SCLK
    //   MISO bit advancement               -> negedge SPI_SCLK
    //
    // SPI Mode 0:
    //
    //   FPGA prepares MISO on falling edge
    //   Raspberry Pi samples on rising edge
    //
    // Protocol:
    //
    //   byte 0 = dummy 0x00
    //   byte 1 = result byte 0
    //   ...
    // ============================================================


    // ============================================================
    // Byte buffers
    // ============================================================

    logic [7:0] tx_current_byte;
    logic       tx_current_valid;

    logic [7:0] tx_next_byte;
    logic       tx_next_valid;

    logic       tx_read_pending;


    // ============================================================
    // Serializer state
    // ============================================================

    logic [2:0] tx_bit_index;

    logic [2:0] tx_dummy_index;
    logic       tx_dummy_phase;

    logic       byte_done_pending;


    // ============================================================
    // TX FIFO read request
    // ============================================================

    always_comb begin

        tx_fifo_rd_en = 1'b0;

        if (
            !spi_cs_n &&
            !tx_fifo_empty &&
            !tx_read_pending &&
            (
                !tx_current_valid ||
                !tx_next_valid
            )
        ) begin

            tx_fifo_rd_en = 1'b1;

        end

    end


    // ============================================================
    // Byte-buffer controller
    //
    // IMPORTANT:
    // The real async_fifo_8bit has a registered synchronous
    // read port clocked on posedge spi_sclk.
    //
    // Therefore all FIFO read and byte-buffer management stays
    // in this posedge block.
    // ============================================================

    always_ff @(
        posedge spi_sclk or
        posedge spi_cs_n or
        negedge rst_n
    ) begin

        if (!rst_n) begin

            tx_current_byte  <= 8'h00;
            tx_current_valid <= 1'b0;

            tx_next_byte     <= 8'h00;
            tx_next_valid    <= 1'b0;

            tx_read_pending  <= 1'b0;

        end

        else if (spi_cs_n) begin

            tx_current_byte  <= 8'h00;
            tx_current_valid <= 1'b0;

            tx_next_byte     <= 8'h00;
            tx_next_valid    <= 1'b0;

            tx_read_pending  <= 1'b0;

        end

        else begin

            // ----------------------------------------------------
            // Previous result byte has now been completely sampled.
            //
            // bit 0 is sampled at this rising edge.
            // Only after that sample do we promote the next byte.
            // ----------------------------------------------------

            if (byte_done_pending) begin

                if (tx_next_valid) begin

                    tx_current_byte  <= tx_next_byte;
                    tx_current_valid <= 1'b1;

                    tx_next_valid <= 1'b0;

                end
                else begin

                    tx_current_valid <= 1'b0;

                end

            end


            // ----------------------------------------------------
            // Complete previous registered FIFO read
            // ----------------------------------------------------

            if (tx_read_pending) begin

                if (
                    byte_done_pending &&
                    tx_next_valid
                ) begin

                    tx_next_byte  <= tx_fifo_data;
                    tx_next_valid <= 1'b1;

                end

                else if (!tx_current_valid) begin

                    tx_current_byte  <= tx_fifo_data;
                    tx_current_valid <= 1'b1;

                end

                else if (!tx_next_valid) begin

                    tx_next_byte  <= tx_fifo_data;
                    tx_next_valid <= 1'b1;

                end


                tx_read_pending <= 1'b0;

            end


            // ----------------------------------------------------
            // Start another synchronous FIFO read
            // ----------------------------------------------------

            if (tx_fifo_rd_en) begin
                tx_read_pending <= 1'b1;
            end

        end

    end


    // ============================================================
    // Mode-0 bit serializer
    //
    // Falling edge prepares the bit that will remain stable until
    // the Raspberry Pi samples it at the next rising edge.
    // ============================================================

    always_ff @(
        negedge spi_sclk or
        posedge spi_cs_n or
        negedge rst_n
    ) begin

        if (!rst_n) begin

            tx_bit_index <= 3'd0;

            tx_dummy_index <= 3'd0;
            tx_dummy_phase <= 1'b1;

            byte_done_pending <= 1'b0;

        end

        else if (spi_cs_n) begin

            tx_bit_index <= 3'd0;

            tx_dummy_index <= 3'd0;
            tx_dummy_phase <= 1'b1;

            byte_done_pending <= 1'b0;

        end

        else begin

            byte_done_pending <= 1'b0;


            // ----------------------------------------------------
            // Dummy / prefetch byte
            // ----------------------------------------------------

            if (tx_dummy_phase) begin

                if (tx_dummy_index == 3'd7) begin

                    tx_dummy_index <= 3'd0;
                    tx_dummy_phase <= 1'b0;

                    tx_bit_index <= 3'd0;

                end
                else begin

                    tx_dummy_index <=
                        tx_dummy_index + 1'b1;

                end

            end


            // ----------------------------------------------------
            // Result stream
            // ----------------------------------------------------

            else if (tx_current_valid) begin

                if (tx_bit_index == 3'd7) begin

                    // Current bit is bit 0.
                    //
                    // Do NOT replace current_byte here.
                    // The master still has to sample bit 0 at the
                    // following rising edge.

                    tx_bit_index <= 3'd0;

                    byte_done_pending <= 1'b1;

                end
                else begin

                    tx_bit_index <=
                        tx_bit_index + 1'b1;

                end

            end

        end

    end


    // ============================================================
    // SPI MISO
    // ============================================================

    always_comb begin

        spi_miso = 1'b0;


        if (!spi_cs_n) begin

            // ----------------------------------------------------
            // First byte = dummy 0x00
            // ----------------------------------------------------

            if (tx_dummy_phase) begin

                spi_miso = 1'b0;

            end


            // ----------------------------------------------------
            // Result stream
            // ----------------------------------------------------

            else if (tx_current_valid) begin

                case (tx_bit_index)

                    3'd0:
                        spi_miso = tx_current_byte[7];

                    3'd1:
                        spi_miso = tx_current_byte[6];

                    3'd2:
                        spi_miso = tx_current_byte[5];

                    3'd3:
                        spi_miso = tx_current_byte[4];

                    3'd4:
                        spi_miso = tx_current_byte[3];

                    3'd5:
                        spi_miso = tx_current_byte[2];

                    3'd6:
                        spi_miso = tx_current_byte[1];

                    3'd7:
                        spi_miso = tx_current_byte[0];

                    default:
                        spi_miso = 1'b0;

                endcase

            end

        end

    end


    // ============================================================
    // TX debug outputs
    // ============================================================

    assign dbg_core_tx_valid =
        core_tx_valid;

    assign dbg_tx_fifo_write =
        core_tx_valid &&
        core_tx_ready;

    assign dbg_tx_fifo_nonempty =
        !tx_fifo_empty;

    assign dbg_tx_current_valid =
        tx_current_valid;


    // ============================================================
    // Simulation-only checks
    // ============================================================

`ifndef SYNTHESIS

    always_ff @(posedge spi_sclk) begin

        if (
            rst_n &&
            spi_fifo_wr_en &&
            rx_fifo_full
        ) begin

            $error(
                "de10_lite_yolov8_top: RX FIFO overflow"
            );

        end

    end


    always_ff @(posedge clk_100) begin

        if (
            rst_n &&
            core_tx_valid &&
            tx_fifo_full
        ) begin

            $error(
                "de10_lite_yolov8_top: TX FIFO overflow"
            );

        end

    end

`endif


endmodule