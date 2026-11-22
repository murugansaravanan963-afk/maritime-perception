`timescale 1ns/1ps

module yolov8_mode_b_empty_frame_tb;

    // ========================================================
    // Clock / Reset
    // ========================================================

    logic clk;
    logic rst_n;

    initial clk = 1'b0;

    always #5 clk = ~clk;   // 100 MHz simulation clock


    // ========================================================
    // DUT byte-stream interface
    // ========================================================

    logic       rx_valid;
    logic       rx_ready;
    logic [7:0] rx_data;

    logic       result_ready;
    logic       result_read_start;

    logic       tx_ready;
    logic       tx_valid;
    logic [7:0] tx_data;
    logic       tx_last;


    // ========================================================
    // DUT
    // ========================================================

    yolov8_mode_b_packet_top dut (
        .clk               (clk),
        .rst_n             (rst_n),

        .rx_valid          (rx_valid),
        .rx_ready          (rx_ready),
        .rx_data           (rx_data),

        .result_ready      (result_ready),
        .result_read_start (result_read_start),

        .tx_ready          (tx_ready),
        .tx_valid          (tx_valid),
        .tx_data           (tx_data),
        .tx_last           (tx_last)
    );


    // ========================================================
    // Test configuration
    // ========================================================

    localparam int FRAME_ID = 123;

    // Empty Mode-B packet:
    //
    // 16-byte header
    // 0-byte payload
    // 2-byte CRC
    //
    // Total = 18 bytes

    logic [7:0] packet [0:17];

    logic [7:0] reply [0:255];

    integer reply_count;
    integer watchdog;

    logic [15:0] crc;


    // ========================================================
    // CRC16 / CCITT-FALSE
    //
    // Polynomial = 0x1021
    // Initial    = 0xFFFF
    // ========================================================

    function automatic [15:0] crc16_byte(
        input [15:0] crc_in,
        input [7:0]  data
    );

        integer i;
        reg [15:0] c;

        begin

            c = crc_in ^ (data << 8);

            for (i = 0; i < 8; i = i + 1) begin

                if (c[15])
                    c = (c << 1) ^ 16'h1021;
                else
                    c = c << 1;

            end

            crc16_byte = c;

        end

    endfunction


    // ========================================================
    // Send one byte using VALID / READY
    // ========================================================

    task automatic send_byte(
        input logic [7:0] value
    );

        begin

            @(negedge clk);

            rx_data  = value;
            rx_valid = 1'b1;

            while (!rx_ready)
                @(negedge clk);

            @(negedge clk);

            rx_valid = 1'b0;
            rx_data  = 8'h00;

        end

    endtask


    // ========================================================
    // DEBUG MONITOR
    //
    // This is the important part of this TB.
    //
    // Expected chain:
    //
    // frame_start
    //      ->
    // packet_done
    //      ->
    // empty_frame_done
    //      ->
    // builder_done
    //      ->
    // result_ready
    //      ->
    // TX bytes
    //
    // ========================================================

    always @(posedge clk) begin

        if (rst_n) begin

            if (dut.fs) begin

                $display(
                    "[DBG] frame_start: ncand=%0d time=%0t",
                    dut.ncand,
                    $time
                );

            end


            if (dut.pdone) begin

                $display(
                    "[DBG] packet_done: ncand=%0d crc_err=%b fmt_err=%b time=%0t",
                    dut.ncand,
                    dut.crc_err,
                    dut.fmt_err,
                    $time
                );

            end


            if (dut.empty_frame_done) begin

                $display(
                    "[DBG] empty_frame_done=1 time=%0t",
                    $time
                );

            end


            if (dut.builder_done) begin

                $display(
                    "[DBG] builder_done=1 n_det=%0d overflow=%b time=%0t",
                    dut.builder_n_det,
                    dut.builder_overflow,
                    $time
                );

            end


            if (result_ready) begin

                $display(
                    "[DBG] result_ready=1 time=%0t",
                    $time
                );

            end


            if (tx_valid && tx_ready) begin

                $display(
                    "[DBG] TX byte[%0d]=%02x last=%b time=%0t",
                    reply_count,
                    tx_data,
                    tx_last,
                    $time
                );

            end

        end

    end


    // ========================================================
    // Main Test
    // ========================================================

    integer i;

    initial begin

        // ----------------------------------------------------
        // Initial values
        // ----------------------------------------------------

        rst_n             = 1'b0;

        rx_valid          = 1'b0;
        rx_data           = 8'h00;

        result_read_start = 1'b0;

        tx_ready          = 1'b1;

        reply_count       = 0;


        // ----------------------------------------------------
        // Reset
        // ----------------------------------------------------

        repeat (5)
            @(posedge clk);

        rst_n = 1'b1;

        repeat (3)
            @(posedge clk);


        // ====================================================
        // Build empty Mode-B packet
        // ====================================================

        // Magic = "Y8PP"

        packet[0] = 8'h59;
        packet[1] = 8'h38;
        packet[2] = 8'h50;
        packet[3] = 8'h50;


        // Version = 1

        packet[4] = 8'h01;


        // Command = 0x03
        // FRAME_CANDIDATES / Mode B

        packet[5] = 8'h03;


        // frame_id = 123 = 0x007B
        // big-endian

        packet[6] = 8'h00;
        packet[7] = 8'h7B;


        // n_candidates = 0

        packet[8] = 8'h00;
        packet[9] = 8'h00;


        // n_classes = 6

        packet[10] = 8'h06;


        // bins = 16

        packet[11] = 8'h10;


        // payload_len = 0

        packet[12] = 8'h00;
        packet[13] = 8'h00;
        packet[14] = 8'h00;
        packet[15] = 8'h00;


        // ----------------------------------------------------
        // Calculate CRC over bytes 0..15
        // ----------------------------------------------------

        crc = 16'hFFFF;

        for (i = 0; i < 16; i = i + 1)
            crc = crc16_byte(
                crc,
                packet[i]
            );


        // CRC big-endian

        packet[16] = crc[15:8];
        packet[17] = crc[7:0];


        // ====================================================
        // Test banner
        // ====================================================

        $display("");
        $display("========================================================");
        $display(" MODE-B V2 EMPTY-FRAME DEBUG TEST");
        $display("========================================================");
        $display(" frame_id      = %0d", FRAME_ID);
        $display(" n_candidates  = 0");
        $display(" payload_len   = 0");
        $display(" input bytes   = 18");
        $display(" input CRC     = 0x%04X", crc);
        $display(" expected reply= 14 bytes");
        $display("--------------------------------------------------------");


        // ====================================================
        // Send packet
        // ====================================================

        for (i = 0; i < 18; i = i + 1) begin

            $display(
                "[TB] Sending byte[%0d] = %02X",
                i,
                packet[i]
            );

            send_byte(
                packet[i]
            );

        end


        $display("");
        $display(
            "[TB] Empty packet completely transmitted at time=%0t",
            $time
        );

        $display(
            "[TB] Waiting for result_ready..."
        );


        // ====================================================
        // Wait for result_ready with watchdog
        // ====================================================

        watchdog = 0;

        while (!result_ready && watchdog < 20000) begin

            @(posedge clk);

            watchdog = watchdog + 1;

        end


        if (!result_ready) begin

            $display("");
            $display("========================================================");
            $display(" EMPTY-FRAME DEBUG TIMEOUT");
            $display("========================================================");

            $display(
                "dut.fs               = %b",
                dut.fs
            );

            $display(
                "dut.ncand            = %0d",
                dut.ncand
            );

            $display(
                "dut.pdone            = %b",
                dut.pdone
            );

            $display(
                "dut.crc_err          = %b",
                dut.crc_err
            );

            $display(
                "dut.fmt_err          = %b",
                dut.fmt_err
            );

            $display(
                "dut.empty_frame_done = %b",
                dut.empty_frame_done
            );

            $display(
                "dut.builder_done     = %b",
                dut.builder_done
            );

            $display(
                "dut.builder_n_det    = %0d",
                dut.builder_n_det
            );

            $display(
                "result_ready         = %b",
                result_ready
            );

            $display("========================================================");
            $display("");

            $fatal(
                1,
                "Timeout waiting for empty-frame result"
            );

        end


        // ====================================================
        // Start reading reply
        // ====================================================

        $display("");
        $display(
            "[TB] result_ready received at time=%0t",
            $time
        );

        @(negedge clk);

        result_read_start = 1'b1;

        @(negedge clk);

        result_read_start = 1'b0;


        // ====================================================
        // Capture reply
        // ====================================================

        reply_count = 0;
        watchdog = 0;

        while (watchdog < 1000) begin

            @(posedge clk);

            watchdog = watchdog + 1;

            if (tx_valid && tx_ready) begin

                reply[reply_count] = tx_data;

                reply_count = reply_count + 1;

                if (tx_last) begin

                    $display(
                        "[TB] tx_last received after %0d bytes",
                        reply_count
                    );

                    break;

                end

            end

        end


        // ====================================================
        // Check reply
        // ====================================================

        $display("");
        $display("========================================================");
        $display(" EMPTY-FRAME REPLY CHECK");
        $display("========================================================");

        $display(
            " reply bytes = %0d",
            reply_count
        );


        if (reply_count >= 4) begin

            $display(
                " status      = 0x%02X",
                reply[0]
            );

            $display(
                " frame_id    = %0d",
                {
                    reply[1],
                    reply[2]
                }
            );

            $display(
                " n_det       = %0d",
                reply[3]
            );

        end


        // ----------------------------------------------------
        // Expected:
        //
        // 12-byte fixed header
        // 0 detections
        // 2-byte CRC
        //
        // = 14 bytes
        // ----------------------------------------------------

        if (reply_count != 14) begin

            $display(
                "[FAIL] Expected 14 reply bytes, got %0d",
                reply_count
            );

            $fatal(
                1,
                "Empty-frame reply length mismatch"
            );

        end


        if (reply[0] != 8'h01) begin

            $display(
                "[FAIL] Expected status=0x01, got 0x%02X",
                reply[0]
            );

            $fatal(
                1,
                "Empty-frame status mismatch"
            );

        end


        if ({
            reply[1],
            reply[2]
        } != FRAME_ID[15:0]) begin

            $display(
                "[FAIL] Expected frame_id=%0d, got %0d",
                FRAME_ID,
                {
                    reply[1],
                    reply[2]
                }
            );

            $fatal(
                1,
                "Empty-frame frame_id mismatch"
            );

        end


        if (reply[3] != 8'h00) begin

            $display(
                "[FAIL] Expected n_det=0, got %0d",
                reply[3]
            );

            $fatal(
                1,
                "Empty-frame n_det mismatch"
            );

        end


        // ====================================================
        // PASS
        // ====================================================

        $display("");
        $display("========================================================");
        $display(" MODE-B V2 EMPTY FRAME : PASS");
        $display("========================================================");
        $display(" status      = 0x%02X", reply[0]);
        $display(
            " frame_id    = %0d",
            {
                reply[1],
                reply[2]
            }
        );
        $display(" n_det       = %0d", reply[3]);
        $display(" reply_bytes = %0d", reply_count);
        $display("========================================================");
        $display("");

        $stop;

    end

endmodule