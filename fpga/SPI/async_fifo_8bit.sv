`timescale 1ns / 1ps

module async_fifo_8bit #(
    parameter integer ADDR_WIDTH = 11
)(
    // ============================================================
    // Write domain
    // ============================================================

    input  logic       wr_clk,
    input  logic       wr_rst_n,

    input  logic [7:0] wr_data,
    input  logic       wr_en,

    output logic       wr_full,


    // ============================================================
    // Read domain
    // ============================================================

    input  logic       rd_clk,
    input  logic       rd_rst_n,

    output logic [7:0] rd_data,
    input  logic       rd_en,

    output logic       rd_empty
);

    // ============================================================
    // FIFO depth
    //
    // ADDR_WIDTH = 11:
    //
    //     DEPTH = 2048 bytes
    //
    // Pointer has one additional wrap bit.
    // ============================================================

    localparam integer DEPTH =
        (1 << ADDR_WIDTH);

    localparam integer PTR_WIDTH =
        ADDR_WIDTH + 1;


    // ============================================================
    // Memory
    //
    // Dual-clock RAM style.
    //
    // Quartus should infer FPGA memory where supported.
    // ============================================================

    (* ramstyle = "M9K" *)
    logic [7:0] mem [0:DEPTH-1];


    // ============================================================
    // Binary / Gray pointers
    // ============================================================

    logic [PTR_WIDTH-1:0] wr_bin;
    logic [PTR_WIDTH-1:0] wr_gray;

    logic [PTR_WIDTH-1:0] rd_bin;
    logic [PTR_WIDTH-1:0] rd_gray;


    logic [PTR_WIDTH-1:0] wr_bin_next;
    logic [PTR_WIDTH-1:0] wr_gray_next;

    logic [PTR_WIDTH-1:0] rd_bin_next;
    logic [PTR_WIDTH-1:0] rd_gray_next;


    // ============================================================
    // Cross-domain synchronized pointers
    // ============================================================

    (* altera_attribute = "-name SYNCHRONIZER_IDENTIFICATION FORCED_IF_ASYNCHRONOUS" *)
    logic [PTR_WIDTH-1:0] rd_gray_sync1;
    logic [PTR_WIDTH-1:0] rd_gray_sync2;


    (* altera_attribute = "-name SYNCHRONIZER_IDENTIFICATION FORCED_IF_ASYNCHRONOUS" *)
    logic [PTR_WIDTH-1:0] wr_gray_sync1;
    logic [PTR_WIDTH-1:0] wr_gray_sync2;


    // ============================================================
    // Write / read fire
    // ============================================================

    wire wr_fire =
        wr_en &&
        !wr_full;


    wire rd_fire =
        rd_en &&
        !rd_empty;


    // ============================================================
    // Next binary pointers
    // ============================================================

    always_comb begin

        wr_bin_next =
            wr_bin;

        if (wr_fire)
            wr_bin_next =
                wr_bin + 1'b1;


        rd_bin_next =
            rd_bin;

        if (rd_fire)
            rd_bin_next =
                rd_bin + 1'b1;

    end


    // ============================================================
    // Binary -> Gray
    //
    // gray = binary XOR (binary >> 1)
    // ============================================================

    always_comb begin

        wr_gray_next =
            (
                wr_bin_next >>
                1
            )
            ^
            wr_bin_next;


        rd_gray_next =
            (
                rd_bin_next >>
                1
            )
            ^
            rd_bin_next;

    end


    // ============================================================
    // EMPTY detection
    //
    // FIFO will be empty after next read when:
    //
    //     next read pointer
    //       ==
    //     synchronized write pointer
    // ============================================================

    wire rd_empty_next =
        (
            rd_gray_next ==
            wr_gray_sync2
        );


    // ============================================================
    // FULL detection
    //
    // Standard asynchronous FIFO Gray-pointer full test.
    //
    // Next write pointer equals synchronized read pointer with
    // the two most-significant Gray bits inverted.
    // ============================================================

    wire wr_full_next =
        (
            wr_gray_next
            ==
            {
                ~rd_gray_sync2[
                    PTR_WIDTH-1:
                    PTR_WIDTH-2
                ],

                rd_gray_sync2[
                    PTR_WIDTH-3:0
                ]
            }
        );


    // ============================================================
    // Synchronize READ pointer into WRITE clock domain
    // ============================================================

    always_ff @(posedge wr_clk or negedge wr_rst_n) begin

        if (!wr_rst_n) begin

            rd_gray_sync1 <= '0;
            rd_gray_sync2 <= '0;

        end
        else begin

            rd_gray_sync1 <=
                rd_gray;

            rd_gray_sync2 <=
                rd_gray_sync1;

        end

    end


    // ============================================================
    // Synchronize WRITE pointer into READ clock domain
    // ============================================================

    always_ff @(posedge rd_clk or negedge rd_rst_n) begin

        if (!rd_rst_n) begin

            wr_gray_sync1 <= '0;
            wr_gray_sync2 <= '0;

        end
        else begin

            wr_gray_sync1 <=
                wr_gray;

            wr_gray_sync2 <=
                wr_gray_sync1;

        end

    end


    // ============================================================
    // WRITE DOMAIN
    // ============================================================

    always_ff @(posedge wr_clk or negedge wr_rst_n) begin

        if (!wr_rst_n) begin

            wr_bin  <= '0;
            wr_gray <= '0;

            wr_full <= 1'b0;

        end
        else begin

            if (wr_fire) begin

                mem[
                    wr_bin[ADDR_WIDTH-1:0]
                ] <=
                    wr_data;

            end


            wr_bin <=
                wr_bin_next;

            wr_gray <=
                wr_gray_next;

            wr_full <=
                wr_full_next;

        end

    end


    // ============================================================
    // READ DOMAIN
    //
    // Registered synchronous read.
    // ============================================================

    always_ff @(posedge rd_clk or negedge rd_rst_n) begin

        if (!rd_rst_n) begin

            rd_bin   <= '0;
            rd_gray  <= '0;

            rd_empty <= 1'b1;

            rd_data  <= 8'h00;

        end
        else begin

            if (rd_fire) begin

                rd_data <=
                    mem[
                        rd_bin[ADDR_WIDTH-1:0]
                    ];

            end


            rd_bin <=
                rd_bin_next;

            rd_gray <=
                rd_gray_next;

            rd_empty <=
                rd_empty_next;

        end

    end


`ifndef SYNTHESIS

    // ============================================================
    // Simulation checks
    // ============================================================

    always_ff @(posedge wr_clk) begin

        if (wr_rst_n) begin

            if (
                wr_en &&
                wr_full
            ) begin

                $warning(
                    "async_fifo_8bit: write attempted while FIFO full"
                );

            end

        end

    end


    always_ff @(posedge rd_clk) begin

        if (rd_rst_n) begin

            if (
                rd_en &&
                rd_empty
            ) begin

                $warning(
                    "async_fifo_8bit: read attempted while FIFO empty"
                );

            end

        end

    end

`endif

endmodule