`timescale 1ns / 1ps

module spi_slave_mode0 (
    // ============================================================
    // SPI physical interface
    //
    // Mode 0:
    // CPOL = 0
    // CPHA = 0
    //
    // MSB first
    // ============================================================

    input  logic spi_sclk,
    input  logic spi_cs_n,
    input  logic spi_mosi,
    output logic spi_miso,

    input  logic rst_n,

    // ============================================================
    // RX debug / byte interface
    // ============================================================

    output logic [7:0] rx_byte,
    output logic       rx_byte_valid,

    // ============================================================
    // RX FIFO interface
    //
    // These signals are intended to connect directly to the
    // write side of the asynchronous RX FIFO.
    //
    // rx_fifo_wr_en is asserted during the clock edge on which
    // the eighth SPI bit completes the byte.
    // ============================================================

    output logic [7:0] rx_fifo_data,
    output logic       rx_fifo_wr_en,

    // ============================================================
    // TX byte interface
    // ============================================================

    input  logic [7:0] tx_byte,
    input  logic       tx_byte_load,

    output logic tx_byte_done
);


    // ============================================================
    // RX shift register
    // ============================================================

    logic [7:0] rx_shift;
    logic [2:0] rx_bit_count;


    // ============================================================
    // Combinational completed-byte value
    //
    // On the eighth rising edge:
    //
    //     rx_shift contains bits [7:1]
    //     spi_mosi contains bit [0]
    //
    // Therefore:
    //
    //     completed byte =
    //         {rx_shift[6:0], spi_mosi}
    // ============================================================

    wire [7:0] rx_completed_byte;

    assign rx_completed_byte =
        {
            rx_shift[6:0],
            spi_mosi
        };


    // ============================================================
    // FIFO write interface
    //
    // IMPORTANT:
    //
    // This is intentionally COMBINATIONAL from the SPI receive
    // state.
    //
    // At the rising edge that receives the eighth bit:
    //
    //     rx_bit_count == 7
    //
    // and therefore rx_fifo_wr_en is already HIGH before the
    // edge is evaluated by the FIFO.
    //
    // The asynchronous FIFO can therefore write the completed
    // byte on THAT SAME spi_sclk rising edge.
    //
    // This eliminates the previous last-byte problem where:
    //
    //     byte completes
    //         ->
    //     registered valid becomes 1
    //         ->
    //     FIFO needs another SCLK rising edge
    //
    // which fails for the final byte of an SPI transaction.
    // ============================================================

    always_comb begin

        rx_fifo_wr_en =
            1'b0;

        rx_fifo_data =
            rx_completed_byte;


        if (
            rst_n &&
            !spi_cs_n &&
            (rx_bit_count == 3'd7)
        ) begin

            rx_fifo_wr_en =
                1'b1;

        end

    end


    // ============================================================
    // RX sequential logic
    //
    // Mode 0:
    //
    // MOSI is sampled on rising edge.
    // ============================================================

    always_ff @(posedge spi_sclk or negedge rst_n) begin

        if (!rst_n) begin

            rx_shift <=
                8'h00;

            rx_bit_count <=
                3'd0;

            rx_byte <=
                8'h00;

            rx_byte_valid <=
                1'b0;

        end

        else begin

            // Debug valid pulse.
            rx_byte_valid <=
                1'b0;


            if (!spi_cs_n) begin

                // ------------------------------------------------
                // Shift incoming MOSI bit
                // ------------------------------------------------

                rx_shift <=
                    {
                        rx_shift[6:0],
                        spi_mosi
                    };


                // ------------------------------------------------
                // Complete byte
                // ------------------------------------------------

                if (
                    rx_bit_count ==
                    3'd7
                ) begin

                    rx_byte <=
                        rx_completed_byte;


                    rx_byte_valid <=
                        1'b1;


                    rx_bit_count <=
                        3'd0;

                end

                else begin

                    rx_bit_count <=
                        rx_bit_count + 1'b1;

                end

            end

            else begin

                // ------------------------------------------------
                // CS high:
                //
                // discard any incomplete partial byte and restore
                // byte alignment.
                // ------------------------------------------------

                rx_bit_count <=
                    3'd0;

            end

        end

    end


    // ============================================================
    // TX datapath
    //
    // Mode 0:
    //
    // Slave updates MISO on falling edge.
    // Master samples MISO on rising edge.
    //
    // MSB first.
    // ============================================================

    logic [7:0] tx_shift;
    logic [2:0] tx_bit_count;
    logic       tx_active;


    always_ff @(negedge spi_sclk or negedge rst_n) begin

        if (!rst_n) begin

            tx_shift <=
                8'h00;

            tx_bit_count <=
                3'd0;

            tx_active <=
                1'b0;

            spi_miso <=
                1'b0;

            tx_byte_done <=
                1'b0;

        end

        else begin

            tx_byte_done <=
                1'b0;


            if (spi_cs_n) begin

                // ------------------------------------------------
                // SPI transaction inactive
                // ------------------------------------------------

                tx_bit_count <=
                    3'd0;

                tx_active <=
                    1'b0;

                spi_miso <=
                    1'b0;


                // ------------------------------------------------
                // Optional preload while CS is high
                // ------------------------------------------------

                if (tx_byte_load) begin

                    tx_shift <=
                        tx_byte;

                    tx_active <=
                        1'b1;

                    spi_miso <=
                        tx_byte[7];

                end

            end

            else begin

                // ------------------------------------------------
                // Active transaction
                // ------------------------------------------------

                if (tx_byte_load) begin

                    tx_shift <=
                        tx_byte;

                    tx_active <=
                        1'b1;

                    tx_bit_count <=
                        3'd0;

                    spi_miso <=
                        tx_byte[7];

                end


                else if (tx_active) begin

                    if (
                        tx_bit_count ==
                        3'd7
                    ) begin

                        // ----------------------------------------
                        // Byte complete
                        // ----------------------------------------

                        tx_byte_done <=
                            1'b1;

                        tx_bit_count <=
                            3'd0;

                        tx_active <=
                            1'b0;

                        spi_miso <=
                            1'b0;

                    end

                    else begin

                        // ----------------------------------------
                        // Shift next bit toward MSB
                        // ----------------------------------------

                        tx_shift <=
                            {
                                tx_shift[6:0],
                                1'b0
                            };


                        spi_miso <=
                            tx_shift[6];


                        tx_bit_count <=
                            tx_bit_count + 1'b1;

                    end

                end


                else begin

                    spi_miso <=
                        1'b0;

                end

            end

        end

    end


`ifndef SYNTHESIS

    // ============================================================
    // Simulation checks
    // ============================================================

    always @(posedge spi_sclk) begin

        if (
            rst_n &&
            spi_cs_n &&
            rx_fifo_wr_en
        ) begin

            $error(
                "spi_slave_mode0: RX FIFO write while CS is high"
            );

        end

    end

`endif

endmodule