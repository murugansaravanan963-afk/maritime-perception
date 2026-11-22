`timescale 1ns / 1ps

module tb_spi_tx_only;

    // ============================================================
    // SPI
    // ============================================================

    logic spi_sclk;
    logic spi_cs_n;
    logic spi_miso;

    // ============================================================
    // Golden result
    // ============================================================

    logic [7:0] golden [0:25];

    // ============================================================
    // Simplified TX byte memory
    //
    // This represents the data that has already been generated
    // by the YOLO result builder.
    // ============================================================

    logic [7:0] tx_mem [0:25];

    integer rd_ptr;

    // ============================================================
    // Serializer
    // ============================================================

    logic [7:0] current_byte;
    logic [2:0] bit_index;

    logic [3:0] dummy_count;

    logic dummy_phase;
    logic current_valid;


    // ============================================================
    // Captured SPI bytes
    // ============================================================

    logic [7:0] rx_byte;

    integer i;
    integer errors;


    // ============================================================
    // Clock
    //
    // 1 MHz SPI
    // ============================================================

    initial begin
        spi_sclk = 1'b0;
    end


    // ============================================================
    // Golden data
    // ============================================================

    initial begin

        golden[0]  = 8'h01;
        golden[1]  = 8'h00;
        golden[2]  = 8'h01;
        golden[3]  = 8'h01;
        golden[4]  = 8'h00;
        golden[5]  = 8'h00;
        golden[6]  = 8'h05;
        golden[7]  = 8'h89;

        golden[8]  = 8'h00;
        golden[9]  = 8'h00;
        golden[10] = 8'h00;
        golden[11] = 8'h59;
        golden[12] = 8'h00;
        golden[13] = 8'h00;
        golden[14] = 8'h00;
        golden[15] = 8'h00;

        golden[16] = 8'h05;
        golden[17] = 8'hA5;
        golden[18] = 8'h05;
        golden[19] = 8'hA5;
        golden[20] = 8'h00;
        golden[21] = 8'hC8;
        golden[22] = 8'h00;
        golden[23] = 8'h00;

        golden[24] = 8'h34;
        golden[25] = 8'hE4;


        for (i = 0; i < 26; i = i + 1)
            tx_mem[i] = golden[i];

    end


    // ============================================================
    // TX state initialization
    // ============================================================

    initial begin

        spi_cs_n = 1'b1;

        rd_ptr = 0;

        current_byte  = 8'h00;
        current_valid = 1'b0;

        bit_index = 3'd0;

        dummy_count = 4'd0;
        dummy_phase = 1'b1;

        errors = 0;

    end


    // ============================================================
    // MODE-0 TX
    //
    // Falling edge prepares the NEXT bit.
    // Rising edge is used only by the master to sample MISO.
    // ============================================================

    always @(negedge spi_sclk) begin

        if (!spi_cs_n) begin

            // ----------------------------------------------------
            // Dummy byte
            // ----------------------------------------------------

            if (dummy_phase) begin

                if (dummy_count == 4'd7) begin

                    dummy_phase = 1'b0;
                    dummy_count = 4'd0;

                    current_byte  = tx_mem[0];
                    current_valid = 1'b1;

                    rd_ptr = 1;

                    bit_index = 3'd0;

                end
                else begin

                    dummy_count = dummy_count + 1;

                end

            end


            // ----------------------------------------------------
            // Result stream
            // ----------------------------------------------------

            else if (current_valid) begin

                if (bit_index == 3'd7) begin

                    bit_index = 3'd0;

                    if (rd_ptr < 26) begin

                        current_byte = tx_mem[rd_ptr];
                        rd_ptr = rd_ptr + 1;

                    end
                    else begin

                        current_valid = 1'b0;

                    end

                end
                else begin

                    bit_index = bit_index + 1'b1;

                end

            end

        end

    end


    // ============================================================
    // MISO
    // ============================================================

    always_comb begin

        spi_miso = 1'b0;

        if (!spi_cs_n) begin

            if (dummy_phase) begin

                spi_miso = 1'b0;

            end
            else if (current_valid) begin

                spi_miso =
                    current_byte[7 - bit_index];

            end

        end

    end


    // ============================================================
    // SPI master byte read
    //
    // Mode 0:
    //
    // Falling edge -> slave prepares
    // Rising edge  -> master samples
    // ============================================================

    task automatic spi_read_byte(
        output logic [7:0] data
    );

        integer b;

        begin

            data = 8'h00;

            for (b = 7; b >= 0; b = b - 1) begin

                // Falling edge
                #500;
                spi_sclk = 1'b0;

                // Rising edge
                #500;
                spi_sclk = 1'b1;

                data[b] = spi_miso;

            end

        end

    endtask


    // ============================================================
    // Main test
    // ============================================================

    initial begin

        logic [7:0] dummy;
        logic [7:0] received;

        #2000;


        // --------------------------------------------------------
        // Start SPI transaction
        // --------------------------------------------------------

        spi_cs_n = 1'b0;


        // --------------------------------------------------------
        // Dummy byte
        // --------------------------------------------------------

        spi_read_byte(dummy);

        $display(
            "Dummy byte = 0x%02X",
            dummy
        );


        if (dummy !== 8'h00) begin

            $display(
                "[ERROR] Dummy byte expected 0x00"
            );

            errors = errors + 1;

        end


        // --------------------------------------------------------
        // 26 result bytes
        // --------------------------------------------------------

        for (i = 0; i < 26; i = i + 1) begin

            spi_read_byte(received);

            if (received === golden[i]) begin

                $display(
                    "[MATCH   ] byte[%02d] = 0x%02X",
                    i,
                    received
                );

            end
            else begin

                $display(
                    "[MISMATCH] byte[%02d] GOLDEN=0x%02X RX=0x%02X",
                    i,
                    golden[i],
                    received
                );

                errors = errors + 1;

            end

        end


        // --------------------------------------------------------
        // End transaction
        // --------------------------------------------------------

        #500;
        spi_cs_n = 1'b1;


        // --------------------------------------------------------
        // Summary
        // --------------------------------------------------------

        $display("");
        $display(
            "============================================================"
        );

        if (errors == 0) begin

            $display(
                "*** TX-ONLY BYTE-EXACT TEST PASSED: 26/26 ***"
            );

        end
        else begin

            $display(
                "*** TX-ONLY TEST FAILED: %0d errors ***",
                errors
            );

        end

        $display(
            "============================================================"
        );


        #1000;
        $finish;

    end

endmodule