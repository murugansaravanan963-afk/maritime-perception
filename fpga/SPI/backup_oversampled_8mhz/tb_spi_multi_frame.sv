`timescale 1ns / 1ps

// ================================================================
// Multi-frame SPI regression (no reset between frames)
//
// Models what the Raspberry Pi actually does:
//   write packet (CS low .. high)
//   wait for result_pending (GPIO)
//   read 1 dummy + N bytes, N may be much longer than the reply
//   -> repeat
//
// Checks per frame:
//   * SPI bytes == bytes the core pushed into the TX FIFO
//   * bytes after the reply read as 0x00
//   * reply CRC16 is self-consistent
//   * status / n_det fields
// Sequence also covers: idle read with nothing pending, 0-candidate
// frame, 20-candidate frame with 1215-byte read, bad-CRC packet,
// recovery afterwards, and a repeated frame giving the same result.
// ================================================================

module tb_spi_multi_frame;

    parameter integer SPI_HALF_NS = 125;      // 4 MHz
    localparam integer MAX_READ   = 1215;     // Pi client worst case reply

    logic clk_100 = 1'b0;
    logic rst_n;
    logic spi_sclk, spi_cs_n, spi_mosi, spi_miso;
    logic result_ready, result_pending;

    always #5 clk_100 = ~clk_100;

    de10_lite_yolov8_top dut (
        .clk_100        (clk_100),
        .rst_n          (rst_n),
        .spi_sclk       (spi_sclk),
        .spi_cs_n       (spi_cs_n),
        .spi_mosi       (spi_mosi),
        .spi_miso       (spi_miso),
        .result_ready   (result_ready),
        .result_pending (result_pending),
        .dbg_core_tx_valid    (),
        .dbg_tx_fifo_write    (),
        .dbg_tx_fifo_nonempty (),
        .dbg_tx_current_valid ()
    );

    // ------------------------------------------------------------
    function automatic [15:0] crc16_next(input [15:0] c_in, input [7:0] d);
        integer i; reg [15:0] c;
        begin
            c = c_in ^ {d, 8'h00};
            for (i = 0; i < 8; i = i + 1)
                c = c[15] ? ((c << 1) ^ 16'h1021) : (c << 1);
            crc16_next = c;
        end
    endfunction

    // ------------------------------------------------------------
    // Core TX monitor (golden = what the core produced)
    // ------------------------------------------------------------
    logic [7:0] golden [$];

    always @(posedge clk_100)
        if (rst_n && dut.core_tx_valid && dut.core_tx_ready)
            golden.push_back(dut.core_tx_data);

`ifdef MF_DEBUG
    always @(posedge clk_100) begin
        if (dut.u_core.fbp)   $display("  t=%0t fbp", $time);
        if (dut.u_core.fs)    $display("  t=%0t frame_start ncand=%0d", $time, dut.u_core.ncand);
        if (dut.u_core.lpp)   $display("  t=%0t last_payload", $time);
        if (dut.u_core.done)  $display("  t=%0t compute done", $time);
        if (dut.u_core.pdone) $display("  t=%0t packet_done fmt=%0d", $time, dut.u_core.fmt_err);
        if (dut.core_result_read_start) $display("  t=%0t tx start", $time);
    end
`endif

    // ------------------------------------------------------------
    // SPI mode-0 byte transfer (master side)
    // ------------------------------------------------------------
    task automatic spi_byte(input logic [7:0] tx, output logic [7:0] rx);
        integer b;
        begin
            for (b = 7; b >= 0; b = b - 1) begin
                spi_mosi = tx[b];
                #(SPI_HALF_NS);
                rx[b] = spi_miso;          // sample just before rising edge
                spi_sclk = 1'b1;
                #(SPI_HALF_NS);
                spi_sclk = 1'b0;
            end
        end
    endtask

    // ------------------------------------------------------------
    // Packet builder
    // ------------------------------------------------------------
    logic [7:0] pkt [$];

    task automatic build_packet(input integer fid, input integer ncand,
                                input integer seed, input bit bad_crc);
        integer c, j, len;
        reg [15:0] crc;
        reg [14:0] anchor;
        begin
            pkt.delete();
            len = ncand * 68;
            pkt.push_back(8'h59); pkt.push_back(8'h38);
            pkt.push_back(8'h50); pkt.push_back(8'h50);
            pkt.push_back(8'h01); pkt.push_back(8'h03);
            pkt.push_back(fid[15:8]);   pkt.push_back(fid[7:0]);
            pkt.push_back(ncand[15:8]); pkt.push_back(ncand[7:0]);
            pkt.push_back(8'h06); pkt.push_back(8'h10);
            pkt.push_back(len[31:24]); pkt.push_back(len[23:16]);
            pkt.push_back(len[15:8]);  pkt.push_back(len[7:0]);

            for (c = 0; c < ncand; c = c + 1) begin
                anchor = (c * 523 + seed * 97) % 14400;
                pkt.push_back({1'b0, anchor[14:8]});
                pkt.push_back(anchor[7:0]);
                pkt.push_back((c + seed) % 6);
                pkt.push_back(8'd120 + ((c * 37 + seed) % 120));
                for (j = 0; j < 64; j = j + 1)
                    pkt.push_back(((j % 16) * (1 + (c + seed) % 3) + c) % 16);
            end

            crc = 16'hFFFF;
            foreach (pkt[k]) crc = crc16_next(crc, pkt[k]);
            if (bad_crc) crc = crc ^ 16'h0001;
            pkt.push_back(crc[15:8]);
            pkt.push_back(crc[7:0]);
        end
    endtask

    // ------------------------------------------------------------
    integer errors = 0;
    logic [7:0] rxbuf [0:MAX_READ];   // [0] = dummy

    task automatic spi_write_packet;
        logic [7:0] dummy;
        begin
            spi_cs_n = 1'b0; #(SPI_HALF_NS);
            foreach (pkt[k]) spi_byte(pkt[k], dummy);
            #(SPI_HALF_NS); spi_cs_n = 1'b1; spi_mosi = 1'b0;
            #(4 * SPI_HALF_NS);
        end
    endtask

    task automatic spi_read(input integer nbytes);
        integer k;
        begin
            spi_cs_n = 1'b0; #(SPI_HALF_NS);
            for (k = 0; k <= nbytes; k = k + 1)   // +1 for the dummy
                spi_byte(8'h00, rxbuf[k]);
            #(SPI_HALF_NS); spi_cs_n = 1'b1;
            #(4 * SPI_HALF_NS);
        end
    endtask

    // Saved content of a reference reply (for the repeat check)
    logic [7:0] ref_reply [$];

    task automatic run_frame(input string name, input integer fid,
                             input integer ncand, input integer seed,
                             input bit bad_crc, input integer read_len,
                             input integer exp_status_bit,  // 0=ok 1=crc_err
                             input bit save_ref, input bit cmp_ref);
        integer k, glen, ndet, t0;
        reg [15:0] crc_c, crc_r;
        integer fe;
        begin
            fe = errors;
            golden.delete();
            build_packet(fid, ncand, seed, bad_crc);
            spi_write_packet();

            t0 = 0;
            while (result_pending !== 1'b1 && t0 < 2_000_000) begin
                #10; t0 = t0 + 10;
            end
            if (result_pending !== 1'b1) begin
                $display("[%s] ERROR: result_pending never went high", name);
                errors = errors + 1;
            end

            spi_read(read_len);
            glen = golden.size();

            if (result_pending !== 1'b0) begin
                $display("[%s] ERROR: result_pending still high after read", name);
                errors = errors + 1;
            end

            if (rxbuf[0] !== 8'h00) begin
                $display("[%s] ERROR: dummy = 0x%02h", name, rxbuf[0]);
                errors = errors + 1;
            end

            for (k = 0; k < read_len; k = k + 1) begin
                if (k < glen) begin
                    if (rxbuf[k+1] !== golden[k]) begin
                        if (errors - fe < 8)
                            $display("[%s] MISMATCH byte[%0d] core=0x%02h spi=0x%02h",
                                     name, k, golden[k], rxbuf[k+1]);
                        errors = errors + 1;
                    end
                end
                else if (rxbuf[k+1] !== 8'h00) begin
                    if (errors - fe < 8)
                        $display("[%s] ERROR: byte[%0d] after reply = 0x%02h (expected 0x00)",
                                 name, k, rxbuf[k+1]);
                    errors = errors + 1;
                end
            end

            // reply self-consistency
            ndet  = rxbuf[4];
            if (glen != 12 + 12 * ndet + 2) begin
                $display("[%s] ERROR: reply length %0d != 14+12*n_det (n_det=%0d)",
                         name, glen, ndet);
                errors = errors + 1;
            end
            crc_c = 16'hFFFF;
            for (k = 0; k < glen - 2; k = k + 1) crc_c = crc16_next(crc_c, rxbuf[k+1]);
            crc_r = {rxbuf[glen-1], rxbuf[glen]};
            if (crc_c !== crc_r) begin
                $display("[%s] ERROR: reply CRC calc=0x%04h rx=0x%04h", name, crc_c, crc_r);
                errors = errors + 1;
            end
            if (rxbuf[1][exp_status_bit] !== 1'b1 || rxbuf[1][7:2] !== 6'd0 && exp_status_bit == 0) begin
                $display("[%s] ERROR: status=0x%02h (expected bit%0d set)",
                         name, rxbuf[1], exp_status_bit);
                errors = errors + 1;
            end
            if ({rxbuf[2], rxbuf[3]} !== fid[15:0]) begin
                $display("[%s] ERROR: frame_id=0x%02h%02h expected 0x%04h",
                         name, rxbuf[2], rxbuf[3], fid[15:0]);
                errors = errors + 1;
            end

            if (save_ref) begin
                ref_reply.delete();
                for (k = 0; k < glen; k = k + 1) ref_reply.push_back(rxbuf[k+1]);
            end
            if (cmp_ref) begin
                if (ref_reply.size() != glen) begin
                    $display("[%s] ERROR: repeat length %0d != ref %0d", name, glen, ref_reply.size());
                    errors = errors + 1;
                end
                else
                    for (k = 0; k < glen - 2; k = k + 1)
                        if (!(k >= 4 && k <= 7) && ref_reply[k] !== rxbuf[k+1]) begin
                            $display("[%s] ERROR: repeat differs at byte[%0d] ref=0x%02h now=0x%02h",
                                     name, k, ref_reply[k], rxbuf[k+1]);
                            errors = errors + 1;
                        end
            end

            $display("[%s] cand=%0d n_det=%0d status=0x%02h reply=%0dB read=%0dB compute=%0d cyc -> %s",
                     name, ncand, ndet, rxbuf[1], glen, read_len,
                     {rxbuf[9], rxbuf[10], rxbuf[11], rxbuf[12]},
                     (errors == fe) ? "OK" : "FAIL");
        end
    endtask

    // ------------------------------------------------------------
    initial begin
        integer k, nz;

        rst_n = 1'b0; spi_sclk = 1'b0; spi_cs_n = 1'b1; spi_mosi = 1'b0;
        #200 rst_n = 1'b1;
        #200;

        $display("=== Multi-frame SPI regression, SPI half period %0d ns ===", SPI_HALF_NS);

        run_frame("F1 1cand      ", 16'h0001,  1, 0, 0,   26, 0, 1, 0);

        // Read with nothing pending: must return zeros and the 0x00
        // bytes on MOSI must not disturb the next packet.
        spi_read(40);
        nz = 0;
        for (k = 0; k <= 40; k = k + 1) if (rxbuf[k] !== 8'h00) nz = nz + 1;
        if (nz != 0) begin
            $display("[IDLE READ     ] ERROR: %0d non-zero bytes", nz);
            errors = errors + 1;
        end
        else $display("[IDLE READ     ] 41 bytes, all 0x00 -> OK");

        run_frame("F2 0cand      ", 16'h0002,  0, 0, 0,   30, 0, 0, 0);
        run_frame("F3 20cand long", 16'h0003, 20, 3, 0, 1215, 0, 0, 0);
        run_frame("F4 repeat F1  ", 16'h0001,  1, 0, 0,   26, 0, 0, 1);
        run_frame("F5 bad CRC    ", 16'h0005,  1, 0, 1,   26, 1, 0, 0);
        run_frame("F6 after error", 16'h0006, 13, 7, 0,  300, 0, 0, 0);

        $display("");
        if (errors == 0) $display("*** MULTI-FRAME SPI REGRESSION PASSED ***");
        else             $display("*** MULTI-FRAME SPI REGRESSION FAILED (%0d errors) ***", errors);
        $finish;
    end

    initial begin
        #200_000_000;
        $display("*** TIMEOUT ***");
        $finish;
    end

endmodule
