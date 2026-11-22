`timescale 1ns / 1ps

module tb_de10_lite_yolov8_spi;

    // ============================================================
    // Configuration
    // ============================================================

    localparam integer INPUT_PACKET_BYTES = 86;

    // More than enough for this one-candidate test.
    localparam integer MAX_RESULT_BYTES = 256;

    // Current test expects the Result Builder to generate
    // no more than 26 bytes for one detection.
    localparam integer SPI_READ_BYTES = 26;

    // SPI half period in ns. 500 = 1 MHz (Pi test speed).
    // The slave oversamples with clk_100, so keep SCLK <= ~8 MHz.
    parameter integer SPI_HALF_NS = 500;


    // ============================================================
    // Clocks / reset
    // ============================================================

    logic clk_100;
    logic rst_n;

    logic spi_sclk;
    logic spi_cs_n;
    logic spi_mosi;
    logic spi_miso;

    logic result_ready;
    logic result_seen = 1'b0;

    always @(posedge clk_100)
        if (result_ready === 1'b1)
            result_seen <= 1'b1;


    // ============================================================
    // DUT
    // ============================================================

    de10_lite_yolov8_top dut (
        .clk_100      (clk_100),
        .rst_n        (rst_n),

        .spi_sclk     (spi_sclk),
        .spi_cs_n     (spi_cs_n),
        .spi_mosi     (spi_mosi),
        .spi_miso     (spi_miso),

        .result_ready (result_ready)
    );


    // ============================================================
    // 100 MHz core clock
    // ============================================================

    initial begin

        clk_100 = 1'b0;

        forever #5
            clk_100 = ~clk_100;

    end


    // ============================================================
    // CRC16-CCITT
    // ============================================================

    function automatic [15:0] crc16_next(
        input [15:0] crc_in,
        input [7:0]  data
    );

        integer i;
        reg [15:0] c;

        begin

            c =
                crc_in ^
                {
                    data,
                    8'h00
                };


            for (
                i = 0;
                i < 8;
                i = i + 1
            ) begin

                if (c[15]) begin

                    c =
                        (c << 1) ^
                        16'h1021;

                end
                else begin

                    c =
                        c << 1;

                end

            end


            crc16_next =
                c;

        end

    endfunction


    // ============================================================
    // Input packet
    // ============================================================

    logic [7:0]
        packet [0:INPUT_PACKET_BYTES-1];

    reg [15:0]
        input_crc;


    // ============================================================
    // GOLDEN STREAM
    //
    // This is captured DIRECTLY from:
    //
    //     dut.core_tx_valid
    //     dut.core_tx_ready
    //     dut.core_tx_data
    //
    // BEFORE the TX async FIFO and SPI serializer.
    //
    // Therefore this is our real transport golden reference.
    // ============================================================

    logic [7:0]
        core_golden [0:MAX_RESULT_BYTES-1];

    integer
        core_golden_count;


    // ============================================================
    // SPI received stream
    // ============================================================

    logic [7:0]
        spi_result [0:MAX_RESULT_BYTES-1];

    integer
        spi_result_count;


    // ============================================================
    // Test status
    // ============================================================

    integer
        compare_count;

    integer
        mismatch_count;

    integer
        error_count;


    // ============================================================
    // Timing
    // ============================================================

    time
        e2e_start_time;

    time
        input_end_time;

    time
        result_ready_time;

    time
        e2e_end_time;


    // ============================================================
    // SPI Mode-0 full-duplex transfer
    //
    // CPOL = 0
    // CPHA = 0
    //
    // 50 MHz:
    //
    //     period = 20 ns
    //
    // MOSI is prepared while SCLK is low.
    //
    // Master samples MISO shortly after rising edge.
    // ============================================================

    task automatic spi_transfer_byte(
        input  logic [7:0] tx,
        output logic [7:0] rx
    );

        integer bit_idx;

        begin

         rx = 8'h00;


         for (
            bit_idx = 7;
            bit_idx >= 0;
            bit_idx = bit_idx - 1
            ) begin

            // ====================================================
            // SPI MODE 0
            //
            // CPOL = 0
            // CPHA = 0
            //
            // MOSI and MISO must already be stable BEFORE the
            // rising edge.
            //
            // The master samples MISO AT the rising edge, not
            // after the slave's posedge sequential logic has
            // already advanced.
            // ====================================================


            // ----------------------------------------------------
            // Clock low phase
            //
            // Prepare MOSI.
            // ----------------------------------------------------

             spi_mosi = tx[bit_idx];


                #SPI_HALF_NS;


            // ----------------------------------------------------
            // Sample the MISO value that is stable immediately
            // BEFORE the rising edge.
            //
            // This models what a real SPI Mode-0 master captures
            // at the rising edge.
            // ----------------------------------------------------

                rx[bit_idx] = spi_miso;


            // ----------------------------------------------------
            // Rising edge
            //
            // FPGA samples MOSI here.
            //
            // Any FPGA state updates caused by this edge must NOT
            // affect the bit that the master has just sampled.
            // ----------------------------------------------------

                spi_sclk = 1'b1;


                #SPI_HALF_NS;


            // ----------------------------------------------------
            // Falling edge
            //
            // Begin next half-cycle.
            // ----------------------------------------------------

                spi_sclk = 1'b0;

            end

        end

        endtask


    // ============================================================
    // Build deterministic one-candidate packet
    // ============================================================

    task automatic build_packet;

        integer i;
        integer p;

        reg [15:0] crc_tmp;

        begin

            // ====================================================
            // HEADER
            // ====================================================

            // Magic "Y8PP"

            packet[0] =
                8'h59;

            packet[1] =
                8'h38;

            packet[2] =
                8'h50;

            packet[3] =
                8'h50;


            // Version

            packet[4] =
                8'h01;


            // Mode B

            packet[5] =
                8'h03;


            // Frame ID

            packet[6] =
                8'h00;

            packet[7] =
                8'h01;


            // Number of candidates = 1

            packet[8] =
                8'h00;

            packet[9] =
                8'h01;


            // Classes = 6

            packet[10] =
                8'h06;


            // DFL bins = 16

            packet[11] =
                8'h10;


            // Payload length = 68

            packet[12] =
                8'h00;

            packet[13] =
                8'h00;

            packet[14] =
                8'h00;

            packet[15] =
                8'h44;


            // ====================================================
            // CANDIDATE
            // ====================================================

            p =
                16;


            // anchor_idx = 0

            packet[p + 0] =
                8'h00;

            packet[p + 1] =
                8'h00;


            // class = 0

            packet[p + 2] =
                8'h00;


            // score = 200

            packet[p + 3] =
                8'hC8;


            // ====================================================
            // Deterministic DFL logits
            //
            // 4 sides x 16 bins
            // ====================================================

            for (
                i = 0;
                i < 64;
                i = i + 1
            ) begin

                packet[p + 4 + i] =
                    i % 16;

            end


            // ====================================================
            // CRC16
            // ====================================================

            crc_tmp =
                16'hFFFF;


            for (
                i = 0;
                i < 84;
                i = i + 1
            ) begin

                crc_tmp =
                    crc16_next(
                        crc_tmp,
                        packet[i]
                    );

            end


            input_crc =
                crc_tmp;


            packet[84] =
                crc_tmp[15:8];


            packet[85] =
                crc_tmp[7:0];

        end

    endtask


    // ============================================================
    // CORE TX GOLDEN MONITOR
    //
    // Capture the byte at the exact ready/valid transfer point.
    //
    // This is BEFORE:
    //
    //     TX async FIFO
    //     SPI serializer
    //
    // So if SPI output differs from this stream, the error is
    // definitely in the transport layer.
    // ============================================================

    always_ff @(posedge clk_100 or negedge rst_n) begin

        if (!rst_n) begin

            core_golden_count <=
                0;

        end
        else begin

            if (
                dut.core_tx_valid &&
                dut.core_tx_ready
            ) begin

                if (
                    core_golden_count <
                    MAX_RESULT_BYTES
                ) begin

                    core_golden[
                        core_golden_count
                    ] <=
                        dut.core_tx_data;


                    $display(
                        "[CORE TX] byte[%0d] = 0x%02h",
                        core_golden_count,
                        dut.core_tx_data
                    );


                    core_golden_count <=
                        core_golden_count + 1;

                end
                else begin

                    $display(
                        "[ERROR] CORE golden buffer overflow"
                    );

                end

            end

        end

    end


    // ============================================================
    // Main test
    // ============================================================

    integer i;

    logic [7:0]
        dummy_rx;

    logic [7:0]
        spi_rx;


    initial begin

        // ========================================================
        // Initial state
        // ========================================================

        rst_n =
            1'b0;


        spi_sclk =
            1'b0;


        spi_cs_n =
            1'b1;


        spi_mosi =
            1'b0;


        spi_result_count =
            0;


        compare_count =
            0;


        mismatch_count =
            0;


        error_count =
            0;


        // ========================================================
        // Reset
        // ========================================================

        #200;


        rst_n =
            1'b1;


        #200;


        // ========================================================
        // Build packet
        // ========================================================

        build_packet();


        // ========================================================
        // Banner
        // ========================================================

        $display("");

        $display(
            "============================================================"
        );

        $display(
            " MODE-B SPI TRANSPORT BYTE-EXACT REGRESSION"
        );

        $display(
            "============================================================"
        );

        $display(
            " Core clock  : 100 MHz"
        );

        $display(
            " SPI half-per: %0d ns", SPI_HALF_NS
        );

        $display(
            " Candidates  : 1"
        );

        $display(
            " Input bytes : %0d",
            INPUT_PACKET_BYTES
        );

        $display(
            " Input CRC16 : 0x%04h",
            input_crc
        );

        $display(
            "============================================================"
        );

        $display("");


        // ========================================================
        // INPUT SPI TRANSACTION
        // ========================================================

        e2e_start_time =
            $time;


        spi_cs_n =
            1'b0;


        #20;


        for (
            i = 0;
            i < INPUT_PACKET_BYTES;
            i = i + 1
        ) begin

            spi_transfer_byte(
                packet[i],
                dummy_rx
            );

        end


        #20;


        spi_cs_n =
            1'b1;


        spi_mosi =
            1'b0;


        input_end_time =
            $time;


        $display(
            "[INFO] Input packet sent at time %0t",
            input_end_time
        );


        // ========================================================
        // WAIT FOR RESULT
        // ========================================================

        // result_ready can pulse BEFORE CS goes high (the core
        // finishes while the CRC bytes are still being clocked),
        // so wait on a sticky flag instead of the live level.
        wait (
            result_seen ===
            1'b1
        );


        result_ready_time =
            $time;


        $display(
            "[INFO] FPGA result_ready at time %0t",
            result_ready_time
        );


        // ========================================================
        // Wait until Result Builder has streamed the result packet
        // into the TX FIFO.
        //
        // For this small one-candidate test, 1 us is generous.
        // ========================================================

        #1000;


        $display(
            "[INFO] Core golden bytes captured = %0d",
            core_golden_count
        );


        if (
            core_golden_count ==
            0
        ) begin

            $display(
                "[ERROR] No native core TX bytes were captured"
            );

            error_count =
                error_count + 1;

        end


        // ========================================================
        // SPI RESULT TRANSACTION
        //
        // First byte = dummy / prefetch.
        // ========================================================

        spi_cs_n =
            1'b0;


        #20;


        spi_transfer_byte(
            8'h00,
            dummy_rx
        );


        $display(
            "[INFO] SPI prefetch dummy = 0x%02h",
            dummy_rx
        );


        // ========================================================
        // Read exactly as many bytes as the native core generated.
        //
        // This is much better than assuming 26 bytes.
        // ========================================================

        spi_result_count =
            0;


        for (
            i = 0;
            i < core_golden_count;
            i = i + 1
        ) begin

            spi_transfer_byte(
                8'h00,
                spi_rx
            );


            spi_result[i] =
                spi_rx;


            $display(
                "[SPI  RX] byte[%0d] = 0x%02h",
                i,
                spi_rx
            );


            spi_result_count =
                spi_result_count + 1;

        end


        #20;


        spi_cs_n =
            1'b1;


        spi_mosi =
            1'b0;


        e2e_end_time =
            $time;


        // ========================================================
        // BYTE-EXACT COMPARISON
        // ========================================================

        $display("");

        $display(
            "============================================================"
        );

        $display(
            " BYTE-EXACT COMPARISON"
        );

        $display(
            "============================================================"
        );


        compare_count =
            0;


        mismatch_count =
            0;


        if (
            spi_result_count !=
            core_golden_count
        ) begin

            $display(
                "[ERROR] Length mismatch: CORE=%0d SPI=%0d",
                core_golden_count,
                spi_result_count
            );


            error_count =
                error_count + 1;

        end


        for (
            i = 0;
            i < core_golden_count;
            i = i + 1
        ) begin

            compare_count =
                compare_count + 1;


            if (
                spi_result[i] !==
                core_golden[i]
            ) begin

                $display(
                    "[MISMATCH] byte[%0d] CORE=0x%02h SPI=0x%02h",
                    i,
                    core_golden[i],
                    spi_result[i]
                );


                mismatch_count =
                    mismatch_count + 1;

            end
            else begin

                $display(
                    "[MATCH   ] byte[%0d] = 0x%02h",
                    i,
                    core_golden[i]
                );

            end

        end


        // ========================================================
        // Summary
        // ========================================================

        $display("");

        $display(
            "============================================================"
        );

        $display(
            " REGRESSION SUMMARY"
        );

        $display(
            "============================================================"
        );


        $display(
            "Core native bytes : %0d",
            core_golden_count
        );


        $display(
            "SPI return bytes  : %0d",
            spi_result_count
        );


        $display(
            "Compared bytes    : %0d",
            compare_count
        );


        $display(
            "Mismatched bytes  : %0d",
            mismatch_count
        );


        $display(
            "Input SPI time    : %0t ns",
            (
                input_end_time -
                e2e_start_time
            ) / 1000
        );


        $display(
            "FPGA ready latency: %0t ns",
            (
                result_ready_time -
                e2e_start_time
            ) / 1000
        );


        $display(
            "Full SPI E2E      : %0t ns",
            (
                e2e_end_time -
                e2e_start_time
            ) / 1000
        );


        // ========================================================
        // Final verdict
        // ========================================================

        if (
            mismatch_count !=
            0
        ) begin

            error_count =
                error_count + 1;

        end


        $display("");


        if (
            (error_count == 0)
            &&
            (core_golden_count > 0)
            &&
            (
                spi_result_count ==
                core_golden_count
            )
            &&
            (
                mismatch_count ==
                0
            )
        ) begin

            $display(
                "*** BYTE-EXACT SPI REGRESSION PASSED ***"
            );

        end
        else begin

            $display(
                "*** BYTE-EXACT SPI REGRESSION FAILED ***"
            );

        end


        $display(
            "============================================================"
        );

        $display("");


        $finish;

    end


    // ============================================================
    // Timeout
    // ============================================================

    initial begin

        #20000000;


        $display("");

        $display(
            "[ERROR] BYTE-EXACT SPI REGRESSION TIMEOUT"
        );


        $finish;

    end

endmodule