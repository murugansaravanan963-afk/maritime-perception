`timescale 1ns / 1ps

module candidate_collector #(
    parameter integer COORD_BITS  = 14,
    parameter integer CLASS_BITS  = 3,
    parameter integer SCORE_BITS  = 8,
    parameter integer ANCHOR_BITS = 15,

    parameter integer MAX_CAND    = 4096
)(
    input logic clk,
    input logic rst_n,

    // ============================================================
    // Frame control
    // ============================================================

    input logic frame_start,
    input logic frame_end,

    // ============================================================
    // Decoded candidate stream
    // ============================================================

    input logic in_valid,

    input logic [COORD_BITS-1:0] x1,
    input logic [COORD_BITS-1:0] y1,
    input logic [COORD_BITS-1:0] x2,
    input logic [COORD_BITS-1:0] y2,

    input logic [CLASS_BITS-1:0] cls,
    input logic [SCORE_BITS-1:0] score,

    input logic pass,

    input logic [ANCHOR_BITS-1:0] anchor_idx,

    // ============================================================
    // Collector status
    // ============================================================

    output logic [$clog2(MAX_CAND+1)-1:0]
        cand_count,

    output logic overflow,

    output logic collect_done,

    // ============================================================
    // Candidate RAM read port
    //
    // Synchronous read:
    //
    // Cycle N:
    //     rd_addr presented
    //
    // Cycle N+1:
    //     rd_* valid
    // ============================================================

    input logic [$clog2(MAX_CAND)-1:0]
        rd_addr,

    output logic [COORD_BITS-1:0] rd_x1,
    output logic [COORD_BITS-1:0] rd_y1,
    output logic [COORD_BITS-1:0] rd_x2,
    output logic [COORD_BITS-1:0] rd_y2,

    output logic [CLASS_BITS-1:0] rd_cls,
    output logic [SCORE_BITS-1:0] rd_score,

    output logic [ANCHOR_BITS-1:0] rd_anchor
);


    // ============================================================
    // Derived widths
    // ============================================================

    localparam integer COUNT_BITS =
        $clog2(MAX_CAND + 1);

    localparam integer ADDR_BITS =
        $clog2(MAX_CAND);


    // ============================================================
    // Packed candidate format
    //
    // MSB
    //
    // | x1 | y1 | x2 | y2 | cls | score | anchor |
    //
    // LSB
    // ============================================================

    localparam integer CAND_WIDTH =
        (4 * COORD_BITS)
        + CLASS_BITS
        + SCORE_BITS
        + ANCHOR_BITS;


    // ============================================================
    // Packed field positions
    // ============================================================

    localparam integer ANCHOR_LSB = 0;

    localparam integer ANCHOR_MSB =
        ANCHOR_LSB
        + ANCHOR_BITS
        - 1;


    localparam integer SCORE_LSB =
        ANCHOR_MSB
        + 1;

    localparam integer SCORE_MSB =
        SCORE_LSB
        + SCORE_BITS
        - 1;


    localparam integer CLS_LSB =
        SCORE_MSB
        + 1;

    localparam integer CLS_MSB =
        CLS_LSB
        + CLASS_BITS
        - 1;


    localparam integer Y2_LSB =
        CLS_MSB
        + 1;

    localparam integer Y2_MSB =
        Y2_LSB
        + COORD_BITS
        - 1;


    localparam integer X2_LSB =
        Y2_MSB
        + 1;

    localparam integer X2_MSB =
        X2_LSB
        + COORD_BITS
        - 1;


    localparam integer Y1_LSB =
        X2_MSB
        + 1;

    localparam integer Y1_MSB =
        Y1_LSB
        + COORD_BITS
        - 1;


    localparam integer X1_LSB =
        Y1_MSB
        + 1;

    localparam integer X1_MSB =
        X1_LSB
        + COORD_BITS
        - 1;


    // ============================================================
    // Candidate RAM
    //
    // Vivado-compatible simple dual-port BRAM inference.
    //
    // Port A:
    //     synchronous write
    //
    // Port B:
    //     synchronous read
    //
    // IMPORTANT:
    //     Do NOT reset or clear this memory.
    //
    // cand_count defines which locations belong to the current
    // frame.
    // ============================================================

    (* ram_style = "block" *)
    logic [CAND_WIDTH-1:0]
        candidate_mem [0:MAX_CAND-1];


    // ============================================================
    // Packed write word
    // ============================================================

    logic [CAND_WIDTH-1:0]
        write_candidate;


    assign write_candidate =
        {
            x1,
            y1,
            x2,
            y2,
            cls,
            score,
            anchor_idx
        };


    // ============================================================
    // Explicit RAM write control
    // ============================================================

    logic
        mem_we;


    logic [ADDR_BITS-1:0]
        mem_wr_addr;


    logic [CAND_WIDTH-1:0]
        mem_wr_data;


    always_comb begin

        mem_we =
            1'b0;


        mem_wr_addr =
            cand_count[
                ADDR_BITS-1:0
            ];


        mem_wr_data =
            write_candidate;


        if (
            !frame_start
            &&
            in_valid
            &&
            pass
            &&
            (
                cand_count <
                MAX_CAND
            )
        ) begin

            mem_we =
                1'b1;

        end

    end


    // ============================================================
    // BRAM read data register
    // ============================================================

    logic [CAND_WIDTH-1:0]
        candidate_read_data;


    // ============================================================
    // RAM inference block
    //
    // Keep BOTH memory accesses in this single clocked process.
    //
    // This matches the standard Vivado simple dual-port BRAM
    // inference template:
    //
    //     one synchronous write port
    //     one synchronous read port
    //
    // Read latency remains exactly one clock.
    // ============================================================

    always @(posedge clk) begin

        // --------------------------------------------------------
        // Write port
        // --------------------------------------------------------

        if (mem_we) begin

            candidate_mem[
                mem_wr_addr
            ] <=
                mem_wr_data;

        end


        // --------------------------------------------------------
        // Synchronous read port
        // --------------------------------------------------------

        candidate_read_data <=
            candidate_mem[
                rd_addr
            ];

    end


    // ============================================================
    // Collector control
    //
    // IMPORTANT:
    //
    // candidate_mem itself is intentionally NOT reset here.
    // ============================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            cand_count <=
                '0;


            overflow <=
                1'b0;


            collect_done <=
                1'b0;

        end

        else begin

            // ----------------------------------------------------
            // collect_done is always a one-clock pulse.
            // ----------------------------------------------------

            collect_done <=
                1'b0;


            // ====================================================
            // New frame
            // ====================================================

            if (frame_start) begin

                cand_count <=
                    '0;


                overflow <=
                    1'b0;

            end

            else begin

                // =================================================
                // Passing candidate
                // =================================================

                if (
                    in_valid &&
                    pass
                ) begin

                    if (
                        cand_count <
                        MAX_CAND
                    ) begin

                        // -----------------------------------------
                        // Actual RAM write is performed by the
                        // dedicated BRAM inference process above.
                        //
                        // Only update logical candidate count here.
                        // -----------------------------------------

                        cand_count <=
                            cand_count
                            + 1'b1;

                    end

                    else begin

                        // -----------------------------------------
                        // Candidate capacity exceeded.
                        // -----------------------------------------

                        overflow <=
                            1'b1;

                    end

                end

            end


            // ====================================================
            // End of frame
            // ====================================================

            if (frame_end) begin

                collect_done <=
                    1'b1;

            end

        end

    end


    // ============================================================
    // Unpack synchronous RAM output
    // ============================================================

    always_comb begin

        rd_anchor =
            candidate_read_data[
                ANCHOR_MSB:
                ANCHOR_LSB
            ];


        rd_score =
            candidate_read_data[
                SCORE_MSB:
                SCORE_LSB
            ];


        rd_cls =
            candidate_read_data[
                CLS_MSB:
                CLS_LSB
            ];


        rd_y2 =
            candidate_read_data[
                Y2_MSB:
                Y2_LSB
            ];


        rd_x2 =
            candidate_read_data[
                X2_MSB:
                X2_LSB
            ];


        rd_y1 =
            candidate_read_data[
                Y1_MSB:
                Y1_LSB
            ];


        rd_x1 =
            candidate_read_data[
                X1_MSB:
                X1_LSB
            ];

    end


    // ============================================================
    // Simulation-only checks
    // ============================================================

`ifndef SYNTHESIS

    initial begin

        if (
            MAX_CAND < 1
        ) begin

            $error(
                "candidate_collector: MAX_CAND must be >= 1"
            );

        end


        if (
            COORD_BITS < 1
        ) begin

            $error(
                "candidate_collector: COORD_BITS must be >= 1"
            );

        end


        if (
            CLASS_BITS < 1
        ) begin

            $error(
                "candidate_collector: CLASS_BITS must be >= 1"
            );

        end


        if (
            SCORE_BITS < 1
        ) begin

            $error(
                "candidate_collector: SCORE_BITS must be >= 1"
            );

        end


        if (
            ANCHOR_BITS < 1
        ) begin

            $error(
                "candidate_collector: ANCHOR_BITS must be >= 1"
            );

        end


        if (
            CAND_WIDTH !=
            (
                (4 * COORD_BITS)
                + CLASS_BITS
                + SCORE_BITS
                + ANCHOR_BITS
            )
        ) begin

            $error(
                "candidate_collector: candidate packed width mismatch"
            );

        end

    end

`endif

endmodule