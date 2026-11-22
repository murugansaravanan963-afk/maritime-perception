`timescale 1ns / 1ps

module tb_spi_tx_fifo;

    localparam integer RESULT_BYTES = 26;


    // ============================================================
    // Clocks / reset
    // ============================================================

    logic clk_100;
    logic spi_sclk;
    logic rst_n;
    logic spi_cs_n;
    logic spi_miso;


    // ============================================================
    // Golden data
    // ============================================================

    logic [7:0] golden [0:RESULT_BYTES-1];


    // ============================================================
    // Real async FIFO
    // ============================================================

    logic [7:0] fifo_wr_data;
    logic       fifo_wr_en;
    logic       fifo_full;

    logic [7:0] fifo_rd_data;
    logic       fifo_rd_en;
    logic       fifo_empty;


    // ============================================================
    // Byte buffer
    //
    // IMPORTANT:
    // These registers are written ONLY by the posedge SPI block.
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

    // Set after bit 0 has been sampled on rising edge.
    logic       byte_done_pending;


    // ============================================================
    // Test variables
    // ============================================================

    integer i;
    integer errors;

    logic [7:0] dummy;
    logic [7:0] received;


    // ============================================================
    // Golden result
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

    end


    // ============================================================
    // 100 MHz core clock
    // ============================================================

    initial begin

        clk_100 = 1'b0;

        forever
            #5 clk_100 = ~clk_100;

    end


    // ============================================================
    // SPI initial state
    // ============================================================

    initial begin

        spi_sclk = 1'b0;

    end


    // ============================================================
    // Real TX async FIFO
    // ============================================================

    async_fifo_8bit #(
        .ADDR_WIDTH(10)
    ) u_tx_fifo (

        .wr_clk   (clk_100),
        .wr_rst_n (rst_n),

        .wr_data  (fifo_wr_data),
        .wr_en    (fifo_wr_en),
        .wr_full  (fifo_full),

        .rd_clk   (spi_sclk),
        .rd_rst_n (rst_n),

        .rd_data  (fifo_rd_data),
        .rd_en    (fifo_rd_en),
        .rd_empty (fifo_empty)
    );


    // ============================================================
    // FIFO read request
    //
    // Keep current + next byte filled whenever possible.
    // ============================================================

    always_comb begin

        fifo_rd_en = 1'b0;

        if (
            !spi_cs_n &&
            !fifo_empty &&
            !tx_read_pending &&
            (
                !tx_current_valid ||
                !tx_next_valid
            )
        ) begin

            fifo_rd_en = 1'b1;

        end

    end


    // ============================================================
    // Byte-buffer controller
    //
    // Runs ONLY on positive SPI edge.
    //
    // Responsibilities:
    //
    // 1. Complete registered FIFO reads.
    // 2. Request another FIFO byte.
    // 3. Promote next byte after the previous byte has REALLY
    //    completed.
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
            // Byte just completed.
            //
            // The master has already sampled bit 0 at THIS
            // rising edge.
            //
            // Now it is safe to promote the next byte.
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
            // Complete previous synchronous FIFO read
            // ----------------------------------------------------

            if (tx_read_pending) begin

                // If the previous byte was just completed and a
                // next byte already exists, promotion above owns
                // current_byte. Therefore place newly returned
                // FIFO data into the next slot.

                if (
                    byte_done_pending &&
                    tx_next_valid
                ) begin

                    tx_next_byte  <= fifo_rd_data;
                    tx_next_valid <= 1'b1;

                end

                else if (!tx_current_valid) begin

                    tx_current_byte  <= fifo_rd_data;
                    tx_current_valid <= 1'b1;

                end

                else if (!tx_next_valid) begin

                    tx_next_byte  <= fifo_rd_data;
                    tx_next_valid <= 1'b1;

                end

                tx_read_pending <= 1'b0;

            end


            // ----------------------------------------------------
            // Start another registered FIFO read
            // ----------------------------------------------------

            if (fifo_rd_en) begin

                tx_read_pending <= 1'b1;

            end

        end

    end


    // ============================================================
    // Serializer state
    //
    // Falling edge prepares the bit that will be sampled at the
    // next rising edge.
    // ============================================================

    always_ff @(
        negedge spi_sclk or
        posedge spi_cs_n or
        negedge rst_n
    ) begin

        if (!rst_n) begin

            tx_bit_index     <= 3'd0;

            tx_dummy_index   <= 3'd0;
            tx_dummy_phase   <= 1'b1;

            byte_done_pending <= 1'b0;

        end

        else if (spi_cs_n) begin

            tx_bit_index     <= 3'd0;

            tx_dummy_index   <= 3'd0;
            tx_dummy_phase   <= 1'b1;

            byte_done_pending <= 1'b0;

        end

        else begin

            // Default: no byte completion waiting.
            byte_done_pending <= 1'b0;


            // ----------------------------------------------------
            // Dummy byte
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
            // Result byte
            // ----------------------------------------------------

            else if (tx_current_valid) begin

                if (tx_bit_index == 3'd7) begin

                    // bit 0 is currently being presented.
                    //
                    // Do NOT change current_byte here.
                    //
                    // The following rising edge samples bit 0.
                    // Byte promotion occurs in the posedge block.

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

            if (tx_dummy_phase) begin

                spi_miso = 1'b0;

            end

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
    // Write one byte into FIFO
    // ============================================================

    task automatic fifo_write_byte(
        input logic [7:0] data
    );

        begin

            @(negedge clk_100);


            while (fifo_full)
                @(negedge clk_100);


            fifo_wr_data = data;
            fifo_wr_en   = 1'b1;


            @(negedge clk_100);

            fifo_wr_en = 1'b0;

        end

    endtask


    // ============================================================
    // SPI Mode-0 byte read
    //
    // 1 MHz:
    //
    // half period = 500 ns
    // ============================================================

    task automatic spi_read_byte(
        output logic [7:0] data
    );

        integer b;

        begin

            data = 8'h00;


            for (b = 7; b >= 0; b = b - 1) begin

                // -----------------------------------------------
                // Falling edge
                //
                // Slave prepares next MISO bit.
                // -----------------------------------------------

                #500;
                spi_sclk = 1'b0;


                // -----------------------------------------------
                // Rising edge
                //
                // Master samples MISO.
                // -----------------------------------------------

                #500;
                spi_sclk = 1'b1;

                #1;

                data[b] = spi_miso;

            end

        end

    endtask


    // ============================================================
    // Main test
    // ============================================================

    initial begin

        // --------------------------------------------------------
        // Initial state
        // --------------------------------------------------------

        rst_n = 1'b0;

        spi_cs_n = 1'b1;

        fifo_wr_data = 8'h00;
        fifo_wr_en   = 1'b0;

        errors = 0;


        // --------------------------------------------------------
        // Reset
        // --------------------------------------------------------

        #200;

        rst_n = 1'b1;


        // --------------------------------------------------------
        // Fill real FIFO
        // --------------------------------------------------------

        $display("");
        $display(
            "Writing 26 golden bytes into real async FIFO..."
        );


        for (
            i = 0;
            i < RESULT_BYTES;
            i = i + 1
        ) begin

            fifo_write_byte(
                golden[i]
            );

        end


        $display(
            "FIFO write complete."
        );


        // --------------------------------------------------------
        // Allow CDC pointer synchronization
        // --------------------------------------------------------

        #5000;


        // --------------------------------------------------------
        // Begin SPI transaction
        // --------------------------------------------------------

        spi_cs_n = 1'b0;


        // --------------------------------------------------------
        // Dummy byte
        // --------------------------------------------------------

        spi_read_byte(dummy);


        $display("");
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

        for (
            i = 0;
            i < RESULT_BYTES;
            i = i + 1
        ) begin

            spi_read_byte(received);


            if (
                received === golden[i]
            ) begin

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
        // End SPI transaction
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
                "*** REAL-FIFO TX TEST PASSED: 26/26 ***"
            );

        end

        else begin

            $display(
                "*** REAL-FIFO TX TEST FAILED: %0d errors ***",
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