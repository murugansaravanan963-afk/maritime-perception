`timescale 1ns / 1ps

// ================================================================
// DE10-Lite YOLOv8 Mode-B top, SCLK-clocked SPI Mode-0 slave
// (target 20-30 MHz SPI), one reply per SPI transaction.
//
// ---------------------------------------------------------------
// Clocking rules (why this version is safe, unlike the first
// dual-edge version):
//
//   * posedge SPI_SCLK : RX shift register ONLY
//   * negedge SPI_SCLK : TX shift register ONLY
//   * NO state is shared between the posedge and negedge logic.
//   * SPI_CS_N high asynchronously resets the per-transaction
//     shift/bit counters.
//   * All SPI <-> clk_100 crossings are byte-level toggle
//     handshakes; the data registers are stable for >= 1 SCLK
//     period whenever the other side samples them.
//
// RX crossing:
//   SCLK posedge completes a byte -> rx_byte_s, flips rx_tog.
//   clk_100 synchronises rx_tog (3 FF) and copies rx_byte_s
//   (stable for 8 SCLK periods) into the RX FIFO.
//
// TX crossing ("hold" register, owned by clk_100):
//   clk_100 writes hold_byte, then flips fill_tog.
//   SCLK negedge synchronises fill_tog (2 FF). At each byte
//   boundary it loads hold_byte if (fill_sync != ack_tog) and
//   flips ack_tog, else it sends 0x00.
//   clk_100 synchronises ack_tog (3 FF) to know the hold is free.
//
// ---------------------------------------------------------------
// Protocol (compatible with the previous write-then-read usage):
//
//   * Every transaction: byte 0 on MISO = dummy 0x00.
//   * If at least one COMPLETE reply is waiting when CS goes low,
//     exactly ONE reply is sent (bytes 1..L), then 0x00.
//   * MOSI bytes are always fed to the packet parser, which
//     ignores everything until "Y8PP". So a transaction can
//     carry packet N on MOSI while it returns reply N-1 on MISO
//     (pipelined / full-duplex mode).
//   * A reply produced while CS is low is kept for the next
//     transaction.
//   * If the host ends a transaction before the whole reply was
//     clocked out, the rest of that reply is discarded so the
//     next transaction starts cleanly.
//   * result_pending = at least one complete reply is waiting.
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
    //   HIGH while at least one complete reply is waiting.
    output logic result_pending,

    // TX-path debug outputs
    output logic dbg_core_tx_valid,
    output logic dbg_tx_fifo_write,
    output logic dbg_tx_fifo_nonempty,
    output logic dbg_tx_current_valid
);

    // ============================================================
    // SCLK DOMAIN: RX (posedge only)
    // ============================================================

    logic [6:0] rx_sh;
    logic [2:0] rx_cnt;
    logic [7:0] rx_byte_s;
    logic       rx_tog;

    // Per-transaction state, cleared while CS is high.
    always_ff @(posedge spi_sclk or posedge spi_cs_n) begin
        if (spi_cs_n) begin
            rx_sh  <= 7'd0;
            rx_cnt <= 3'd0;
        end
        else begin
            rx_sh  <= {rx_sh[5:0], spi_mosi};
            rx_cnt <= rx_cnt + 3'd1;
        end
    end

    // Completed byte + toggle, persistent across transactions.
    always_ff @(posedge spi_sclk or negedge rst_n) begin
        if (!rst_n) begin
            rx_byte_s <= 8'h00;
            rx_tog    <= 1'b0;
        end
        else if (!spi_cs_n && rx_cnt == 3'd7) begin
            rx_byte_s <= {rx_sh, spi_mosi};
            rx_tog    <= ~rx_tog;
        end
    end


    // ============================================================
    // SCLK DOMAIN: TX (negedge only)
    // ============================================================

    logic [7:0] hold_byte;      // clk_100 domain register
    logic       fill_tog;       // clk_100 domain register

    logic       fill_s1, fill_s2;
    logic       ack_tog;
    logic [7:0] tx_sh;
    logic [2:0] tx_cnt;

    wire hold_valid_s = (fill_s2 != ack_tog);

    // fill_tog synchroniser into the negedge-SCLK domain.
    always_ff @(negedge spi_sclk or negedge rst_n) begin
        if (!rst_n) begin
            fill_s1 <= 1'b0;
            fill_s2 <= 1'b0;
        end
        else begin
            fill_s1 <= fill_tog;
            fill_s2 <= fill_s1;
        end
    end

    // Shift register. CS high -> preload the dummy byte 0x00.
    // Negedge k follows the master's k-th sampling edge, so
    // negedges 1..7 shift and negedge 8 loads the next byte.
    always_ff @(negedge spi_sclk or posedge spi_cs_n) begin
        if (spi_cs_n) begin
            tx_sh  <= 8'h00;
            tx_cnt <= 3'd0;
        end
        else begin
            tx_cnt <= tx_cnt + 3'd1;
            if (tx_cnt == 3'd7)
                tx_sh <= hold_valid_s ? hold_byte : 8'h00;
            else
                tx_sh <= {tx_sh[6:0], 1'b0};
        end
    end

    always_ff @(negedge spi_sclk or negedge rst_n) begin
        if (!rst_n)
            ack_tog <= 1'b0;
        else if (!spi_cs_n && tx_cnt == 3'd7 && hold_valid_s)
            ack_tog <= ~ack_tog;
    end

    assign spi_miso = !spi_cs_n ? tx_sh[7] : 1'b0;


    // ============================================================
    // clk_100 DOMAIN: synchronisers
    // ============================================================

    (* altera_attribute = "-name SYNCHRONIZER_IDENTIFICATION FORCED_IF_ASYNCHRONOUS" *)
    logic [2:0] cs_sync;
    (* altera_attribute = "-name SYNCHRONIZER_IDENTIFICATION FORCED_IF_ASYNCHRONOUS" *)
    logic [2:0] rx_tog_sync;
    (* altera_attribute = "-name SYNCHRONIZER_IDENTIFICATION FORCED_IF_ASYNCHRONOUS" *)
    logic [2:0] ack_sync;

    always_ff @(posedge clk_100 or negedge rst_n) begin
        if (!rst_n) begin
            cs_sync     <= 3'b111;
            rx_tog_sync <= 3'b000;
            ack_sync    <= 3'b000;
        end
        else begin
            cs_sync     <= {cs_sync[1:0],     spi_cs_n};
            rx_tog_sync <= {rx_tog_sync[1:0], rx_tog};
            ack_sync    <= {ack_sync[1:0],    ack_tog};
        end
    end

    wire spi_active = !cs_sync[1];
    wire rx_new     = rx_tog_sync[1] ^ rx_tog_sync[2];
    wire hold_empty = (fill_tog == ack_sync[1]);


    // ============================================================
    // RX: synchronised byte -> RX FIFO -> core
    // ============================================================

    logic [7:0] spi_fifo_data;
    logic       spi_fifo_wr_en;

    always_ff @(posedge clk_100 or negedge rst_n) begin
        if (!rst_n) begin
            spi_fifo_data  <= 8'h00;
            spi_fifo_wr_en <= 1'b0;
        end
        else begin
            spi_fifo_wr_en <= rx_new;
            if (rx_new)
                spi_fifo_data <= rx_byte_s;   // stable for 8 SCLK periods
        end
    end

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

    logic       core_rx_valid;
    logic [7:0] core_rx_data;
    logic       core_rx_ready;
    logic       rx_read_pending;

    assign rx_fifo_rd_en = !rx_fifo_empty && !rx_read_pending && !core_rx_valid;

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
    // Mode-B core
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
    // TX FIFO: 9 bits = {last, data}, 2048 deep (> 1214 B max reply)
    // ============================================================

    logic       tx_fifo_full, tx_fifo_empty, tx_fifo_rd_en;
    logic [8:0] tx_fifo_q;

    wire tx_fifo_wr = core_tx_valid && core_tx_ready;

    tx_sync_fifo9 #(
        .AW(11)
    ) u_tx_fifo (
        .clk     (clk_100),
        .rst_n   (rst_n),
        .wr_en   (tx_fifo_wr),
        .wr_data ({core_tx_last, core_tx_data}),
        .full    (tx_fifo_full),
        .rd_en   (tx_fifo_rd_en),
        .rd_data (tx_fifo_q),
        .empty   (tx_fifo_empty)
    );

    assign core_tx_ready = !tx_fifo_full;


    // ============================================================
    // TX controller (clk_100): one reply per SPI transaction
    // ============================================================

    typedef enum logic [1:0] { TX_IDLE, TX_SERVE, TX_DRAIN } txc_t;
    txc_t txc;

    logic [7:0] pending_count;     // complete replies in the FIFO
    logic       txn_handled;       // CS-low already evaluated
    logic       rd_inflight;       // FIFO read issued, data next cycle
    logic       fed_last;          // last byte of the reply is in hold
    logic [3:0] idle_settle;       // CS-high settle time before retract

    logic spi_active_d;
    wire  cs_fall = spi_active && !spi_active_d;

    // A reply is only started at the CS falling edge, and only if the
    // controller is idle with an empty hold register and CS was high
    // long enough (>= 16 cycles) for everything to settle. Otherwise
    // this transaction returns 0x00 and the reply waits for the next.
    wire reply_queued = tx_fifo_wr && core_tx_last;
    wire serve_start  = (txc == TX_IDLE) && cs_fall && !txn_handled &&
                        (pending_count != 8'd0) &&
                        hold_empty && (idle_settle == 4'd15);

    assign tx_fifo_rd_en =
        !tx_fifo_empty && !rd_inflight &&
        (
          ((txc == TX_SERVE) && spi_active && hold_empty && !fed_last) ||
          (txc == TX_DRAIN)
        );

    always_ff @(posedge clk_100 or negedge rst_n) begin
        if (!rst_n) begin
            txc           <= TX_IDLE;
            pending_count <= 8'd0;
            spi_active_d  <= 1'b0;
            txn_handled   <= 1'b0;
            rd_inflight   <= 1'b0;
            fed_last      <= 1'b0;
            idle_settle   <= 4'd0;
            hold_byte     <= 8'h00;
            fill_tog      <= 1'b0;
        end
        else begin
            // ---- reply accounting ----
            case ({reply_queued, serve_start})
                2'b10:   pending_count <= pending_count + 8'd1;
                2'b01:   pending_count <= pending_count - 8'd1;
                default: ;
            endcase

            spi_active_d <= spi_active;

            // Each CS-low period gets exactly one serve decision.
            if (!spi_active)
                txn_handled <= 1'b0;
            else if (cs_fall)
                txn_handled <= 1'b1;

            if (tx_fifo_rd_en)
                rd_inflight <= 1'b1;

            case (txc)
                // ------------------------------------------------
                TX_IDLE: begin
                    if (spi_active) begin
                        idle_settle <= 4'd0;
                        if (serve_start) begin
                            fed_last <= 1'b0;
                            txc      <= TX_SERVE;
                        end
                    end
                    else begin
                        // CS high: once everything has settled, take
                        // back a hold byte the host never clocked out.
                        if (idle_settle != 4'd15)
                            idle_settle <= idle_settle + 4'd1;
                        else if (!hold_empty)
                            fill_tog <= ~fill_tog;
                    end
                end

                // ------------------------------------------------
                TX_SERVE: begin
                    if (rd_inflight) begin
                        rd_inflight <= 1'b0;
                        if (spi_active) begin
                            hold_byte <= tx_fifo_q[7:0];
                            fill_tog  <= ~fill_tog;
                            fed_last  <= tx_fifo_q[8];
                        end
                        else if (!tx_fifo_q[8]) begin
                            txc <= TX_DRAIN;   // host stopped early
                        end
                        else begin
                            txc         <= TX_IDLE;
                            idle_settle <= 4'd0;
                        end
                    end
                    else if (!spi_active) begin
                        idle_settle <= 4'd0;
                        txc <= fed_last ? TX_IDLE : TX_DRAIN;
                    end
                end

                // ------------------------------------------------
                // Discard the rest of a reply the host did not read.
                TX_DRAIN: begin
                    if (rd_inflight) begin
                        rd_inflight <= 1'b0;
                        if (tx_fifo_q[8]) begin
                            txc         <= TX_IDLE;
                            idle_settle <= 4'd0;
                        end
                    end
                end

                default: txc <= TX_IDLE;
            endcase
        end
    end

    // HIGH while a complete reply is waiting for the host.
    always_ff @(posedge clk_100 or negedge rst_n) begin
        if (!rst_n) result_pending <= 1'b0;
        else        result_pending <= (pending_count != 8'd0);
    end


    // ============================================================
    // Debug outputs (all clk_100 domain)
    // ============================================================

    assign dbg_core_tx_valid    = core_tx_valid;
    assign dbg_tx_fifo_write    = tx_fifo_wr;
    assign dbg_tx_fifo_nonempty = !tx_fifo_empty;
    assign dbg_tx_current_valid = (txc == TX_SERVE);


`ifndef SYNTHESIS
    always_ff @(posedge clk_100) begin
        if (rst_n && spi_fifo_wr_en && rx_fifo_full)
            $error("de10_lite_yolov8_top: RX FIFO overflow");
    end
`endif

endmodule


// ================================================================
// Simple synchronous FIFO, 9-bit wide, registered read port
// (rd_data valid the cycle after rd_en). Infers M9K.
// ================================================================

module tx_sync_fifo9 #(
    parameter integer AW = 11
)(
    input  logic       clk,
    input  logic       rst_n,
    input  logic       wr_en,
    input  logic [8:0] wr_data,
    output logic       full,
    input  logic       rd_en,
    output logic [8:0] rd_data,
    output logic       empty
);
    (* ramstyle = "M9K" *) logic [8:0] mem [0:(1<<AW)-1];
    logic [AW:0] wptr, rptr;

    assign empty = (wptr == rptr);
    assign full  = (wptr[AW] != rptr[AW]) && (wptr[AW-1:0] == rptr[AW-1:0]);

    always_ff @(posedge clk) begin
        if (wr_en && !full)
            mem[wptr[AW-1:0]] <= wr_data;
        if (rd_en && !empty)
            rd_data <= mem[rptr[AW-1:0]];
    end

    always_ff @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            wptr <= '0;
            rptr <= '0;
        end
        else begin
            if (wr_en && !full)  wptr <= wptr + 1'b1;
            if (rd_en && !empty) rptr <= rptr + 1'b1;
        end
    end
endmodule
