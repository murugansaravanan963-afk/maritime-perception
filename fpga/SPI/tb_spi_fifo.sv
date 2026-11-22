`timescale 1ns / 1ps

module tb_spi_fifo;

    // ============================================================
    // Clocks / reset
    // ============================================================

    logic clk_100;
    logic rst_n;

    logic spi_sclk;
    logic spi_cs_n;
    logic spi_mosi;
    logic spi_miso;


    // ============================================================
    // SPI RX debug interface
    // ============================================================

    logic [7:0] spi_rx_byte;
    logic       spi_rx_valid;


    // ============================================================
    // SPI -> FIFO direct interface
    // ============================================================

    logic [7:0] spi_fifo_data;
    logic       spi_fifo_wr_en;


    // ============================================================
    // SPI TX
    // ============================================================

    logic [7:0] spi_tx_byte;
    logic       spi_tx_load;
    logic       spi_tx_done;


    // ============================================================
    // FIFO
    // ============================================================

    logic       fifo_full;

    logic [7:0] fifo_rd_data;
    wire        fifo_rd_en;
    logic       fifo_empty;


    // ============================================================
    // Checker
    // ============================================================

    integer rx_count;
    integer error_count;

    logic [7:0] expected_byte;

    logic rd_valid_d;


    // ============================================================
    // 100 MHz core clock
    // ============================================================

    initial begin

        clk_100 =
            1'b0;

        forever #5
            clk_100 =
                ~clk_100;

    end


    // ============================================================
    // SPI slave
    // ============================================================

    spi_slave_mode0 u_spi (
        .spi_sclk      (spi_sclk),
        .spi_cs_n      (spi_cs_n),
        .spi_mosi      (spi_mosi),
        .spi_miso      (spi_miso),

        .rst_n         (rst_n),

        .rx_byte       (spi_rx_byte),
        .rx_byte_valid (spi_rx_valid),

        .rx_fifo_data  (spi_fifo_data),
        .rx_fifo_wr_en (spi_fifo_wr_en),

        .tx_byte       (spi_tx_byte),
        .tx_byte_load  (spi_tx_load),

        .tx_byte_done  (spi_tx_done)
    );


    // ============================================================
    // RX asynchronous FIFO
    // ============================================================

    async_fifo_8bit #(
        .ADDR_WIDTH(4)
    ) u_rx_fifo (
        .wr_clk   (spi_sclk),
        .wr_rst_n (rst_n),

        .wr_data  (spi_fifo_data),
        .wr_en    (spi_fifo_wr_en),

        .wr_full  (fifo_full),

        .rd_clk   (clk_100),
        .rd_rst_n (rst_n),

        .rd_data  (fifo_rd_data),
        .rd_en    (fifo_rd_en),

        .rd_empty (fifo_empty)
    );


    // ============================================================
    // Read continuously whenever FIFO contains data
    // ============================================================

    assign fifo_rd_en =
        !fifo_empty;


    // ============================================================
    // Raspberry Pi SPI Mode-0 master model
    //
    // 50 MHz
    // ============================================================

    task automatic spi_send_byte(
        input logic [7:0] data
    );

        integer bit_idx;

        begin

            for (
                bit_idx = 7;
                bit_idx >= 0;
                bit_idx = bit_idx - 1
            ) begin

                // MOSI valid before rising edge.
                spi_mosi =
                    data[bit_idx];


                #10;


                // Rising edge:
                // slave samples MOSI.
                spi_sclk =
                    1'b1;


                #10;


                // Falling edge.
                spi_sclk =
                    1'b0;

            end

        end

    endtask


    // ============================================================
    // FIFO checker
    // ============================================================

    always_ff @(posedge clk_100 or negedge rst_n) begin

        if (!rst_n) begin

            rd_valid_d <=
                1'b0;

            rx_count <=
                0;

            error_count <=
                0;

            expected_byte <=
                8'h10;

        end

        else begin

            // ----------------------------------------------------
            // Check previous synchronous FIFO read
            // ----------------------------------------------------

            if (rd_valid_d) begin

                if (
                    fifo_rd_data !==
                    expected_byte
                ) begin

                    $display(
                        "[ERROR] time=%0t expected=0x%02h received=0x%02h",
                        $time,
                        expected_byte,
                        fifo_rd_data
                    );


                    error_count <=
                        error_count + 1;

                end

                else begin

                    $display(
                        "[PASS ] time=%0t byte[%0d] = 0x%02h",
                        $time,
                        rx_count,
                        fifo_rd_data
                    );

                end


                rx_count <=
                    rx_count + 1;


                expected_byte <=
                    expected_byte + 1'b1;

            end


            rd_valid_d <=
                fifo_rd_en &&
                !fifo_empty;

        end

    end


    // ============================================================
    // SPI debug
    // ============================================================

    always @(posedge spi_sclk) begin

        if (
            rst_n &&
            spi_rx_valid
        ) begin

            $display(
                "[SPI RX] time=%0t byte=0x%02h",
                $time,
                spi_rx_byte
            );

        end

    end


    // ============================================================
    // FIFO overflow
    // ============================================================

    always @(posedge spi_sclk) begin

        if (
            rst_n &&
            fifo_full
        ) begin

            $display(
                "[ERROR] RX FIFO FULL at time=%0t",
                $time
            );

        end

    end


    // ============================================================
    // Main test
    // ============================================================

    integer i;


    initial begin

        rst_n =
            1'b0;

        spi_sclk =
            1'b0;

        spi_cs_n =
            1'b1;

        spi_mosi =
            1'b0;

        spi_tx_byte =
            8'h00;

        spi_tx_load =
            1'b0;


        // --------------------------------------------------------
        // Reset
        // --------------------------------------------------------

        #100;

        rst_n =
            1'b1;

        #100;


        // --------------------------------------------------------
        // Banner
        // --------------------------------------------------------

        $display("");
        $display(
            "=============================================="
        );

        $display(
            " SPI + ASYNC FIFO JOINT TEST V2"
        );

        $display(
            " SPI MODE 0"
        );

        $display(
            " SPI SCLK = 50 MHz"
        );

        $display(
            " CORE CLK = 100 MHz"
        );

        $display(
            " TEST DATA = 0x10 ... 0x1F"
        );

        $display(
            "=============================================="
        );

        $display("");


        // --------------------------------------------------------
        // Begin transaction
        // --------------------------------------------------------

        spi_cs_n =
            1'b0;

        #20;


        // --------------------------------------------------------
        // Send 16 bytes
        // --------------------------------------------------------

        for (
            i = 0;
            i < 16;
            i = i + 1
        ) begin

            spi_send_byte(
                8'h10 + i
            );

        end


        // --------------------------------------------------------
        // IMPORTANT:
        //
        // No additional SCLK edge is generated here.
        //
        // This intentionally verifies that the final byte is
        // committed to the FIFO on its own eighth rising edge.
        // --------------------------------------------------------

        #20;

        spi_cs_n =
            1'b1;

        spi_mosi =
            1'b0;


        // --------------------------------------------------------
        // Allow CDC / FIFO to drain
        // --------------------------------------------------------

        #1000;


        // --------------------------------------------------------
        // Summary
        // --------------------------------------------------------

        $display("");
        $display(
            "=============================================="
        );

        $display(
            " TEST SUMMARY"
        );

        $display(
            "=============================================="
        );


        $display(
            "Expected bytes : 16"
        );


        $display(
            "Received bytes : %0d",
            rx_count
        );


        $display(
            "Errors         : %0d",
            error_count
        );


        if (
            (rx_count == 16)
            &&
            (error_count == 0)
        ) begin

            $display("");

            $display(
                "*** SPI + FIFO TEST PASSED ***"
            );

        end

        else begin

            $display("");

            $display(
                "*** SPI + FIFO TEST FAILED ***"
            );

        end


        $display(
            "=============================================="
        );

        $display("");


        $finish;

    end


    // ============================================================
    // Timeout
    // ============================================================

    initial begin

        #20000;

        $display(
            "[ERROR] Simulation timeout"
        );

        $finish;

    end

endmodule