`timescale 1ns / 1ps

module yolov8_postprocess_benchmark_top
(
    input  logic clk,
    input  logic rst_n,

    // Keep one observable output so that the complete datapath
    // cannot be optimized away.
    output logic activity_out
);

    `include "rtl_params.vh"


    // ============================================================
    // Benchmark purpose
    //
    // This module is NOT the final board-level interface.
    //
    // It exists only to:
    //
    //   1. Internalize the very wide simulation interface.
    //   2. Preserve the complete post-processing datapath.
    //   3. Allow Vivado placement and routing.
    //   4. Measure realistic resource usage and timing.
    //
    // Final hardware will later replace this wrapper with SPI.
    // ============================================================


    // ============================================================
    // Internal DUT inputs
    // ============================================================

    logic frame_start;

    logic in_valid;
    logic in_ready;

    logic [ANCHOR_BITS-1:0]
        anchor_idx;

    logic [Q_BITS-1:0]
        bbox_q [0:4*DFL_BINS-1];

    logic [SCORE_Q_BITS-1:0]
        score_q [0:NUM_CLASSES-1];


    // ============================================================
    // DUT outputs
    // ============================================================

    logic det_valid;

    logic [COORD_BITS-1:0]
        det_x1;

    logic [COORD_BITS-1:0]
        det_y1;

    logic [COORD_BITS-1:0]
        det_x2;

    logic [COORD_BITS-1:0]
        det_y2;

    logic [CLASS_BITS-1:0]
        det_cls;

    logic [SCORE_Q_BITS-1:0]
        det_score;

    logic [ANCHOR_BITS-1:0]
        det_anchor_idx;

    logic [$clog2(MAX_DET+1)-1:0]
        n_det;

    logic overflow;
    logic done;


    // ============================================================
    // Internal pseudo-random stimulus
    //
    // The LFSRs prevent synthesis from constant-propagating the
    // entire input datapath away.
    // ============================================================

    logic [31:0]
        lfsr_a;

    logic [31:0]
        lfsr_b;

    logic [31:0]
        lfsr_c;


    wire feedback_a =
        lfsr_a[31]
        ^ lfsr_a[21]
        ^ lfsr_a[1]
        ^ lfsr_a[0];


    wire feedback_b =
        lfsr_b[31]
        ^ lfsr_b[6]
        ^ lfsr_b[4]
        ^ lfsr_b[2];


    wire feedback_c =
        lfsr_c[31]
        ^ lfsr_c[22]
        ^ lfsr_c[2]
        ^ lfsr_c[1];


    // ============================================================
    // Anchor/frame generator
    // ============================================================

    logic [ANCHOR_BITS-1:0]
        anchor_counter;

    logic
        frame_active;


    // ============================================================
    // Input arrays
    // ============================================================

    integer i;


    always_comb begin

        // --------------------------------------------------------
        // Generate different dynamic values for every DFL bin.
        //
        // These values are NOT intended to represent a real frame.
        // They only preserve the actual arithmetic datapath for
        // implementation/timing analysis.
        // --------------------------------------------------------

        for (
            i = 0;
            i < 4 * DFL_BINS;
            i = i + 1
        ) begin

            bbox_q[i] =
                lfsr_a[
                    (i % 4) * 8
                    +: 8
                ]
                ^
                lfsr_b[
                    ((i + 1) % 4) * 8
                    +: 8
                ]
                ^
                i[7:0];

        end


        // --------------------------------------------------------
        // Dynamic class scores
        // --------------------------------------------------------

        for (
            i = 0;
            i < NUM_CLASSES;
            i = i + 1
        ) begin

            score_q[i] =
                lfsr_b[
                    (i % 4) * 8
                    +: 8
                ]
                ^
                lfsr_c[
                    ((i + 2) % 4) * 8
                    +: 8
                ]
                ^
                i[7:0];

        end

    end


    // ============================================================
    // Benchmark stimulus controller
    // ============================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            lfsr_a <=
                32'h1ACE_B00C;

            lfsr_b <=
                32'h1357_9BDF;

            lfsr_c <=
                32'h2468_ACE1;


            anchor_counter <=
                '0;


            anchor_idx <=
                '0;


            frame_start <=
                1'b0;


            frame_active <=
                1'b0;


            in_valid <=
                1'b0;

        end

        else begin

            // ====================================================
            // Continuously evolve stimulus.
            // ====================================================

            lfsr_a <=
                {
                    lfsr_a[30:0],
                    feedback_a
                };


            lfsr_b <=
                {
                    lfsr_b[30:0],
                    feedback_b
                };


            lfsr_c <=
                {
                    lfsr_c[30:0],
                    feedback_c
                };


            // ----------------------------------------------------
            // Default pulse values
            // ----------------------------------------------------

            frame_start <=
                1'b0;


            // ====================================================
            // Start benchmark frame
            // ====================================================

            if (!frame_active) begin

                frame_start <=
                    1'b1;


                frame_active <=
                    1'b1;


                anchor_counter <=
                    '0;


                anchor_idx <=
                    '0;


                in_valid <=
                    1'b0;

            end


            // ====================================================
            // Feed anchors using DUT valid/ready handshake.
            // ====================================================

            else begin

                if (!in_valid) begin

                    in_valid <=
                        1'b1;


                    anchor_idx <=
                        anchor_counter;

                end


                else if (
                    in_valid &&
                    in_ready
                ) begin

                    in_valid <=
                        1'b0;


                    // --------------------------------------------
                    // Last anchor of the frame
                    // --------------------------------------------

                    if (
                        anchor_counter ==
                        N_ANCHORS - 1
                    ) begin

                        anchor_counter <=
                            '0;

                    end


                    else begin

                        anchor_counter <=
                            anchor_counter
                            + 1'b1;

                    end

                end


                // ------------------------------------------------
                // Once the complete postprocess reports done,
                // begin another benchmark frame.
                // ------------------------------------------------

                if (done) begin

                    frame_active <=
                        1'b0;


                    in_valid <=
                        1'b0;

                end

            end

        end

    end


    // ============================================================
    // V4 post-processing DUT
    // ============================================================

    yolov8_postprocess_wrapper u_postprocess_wrapper (
        .clk(
            clk
        ),

        .rst_n(
            rst_n
        ),

        .frame_start(
            frame_start
        ),

        .in_valid(
            in_valid
        ),

        .in_ready(
            in_ready
        ),

        .anchor_idx(
            anchor_idx
        ),

        .bbox_q(
            bbox_q
        ),

        .score_q(
            score_q
        ),

        .det_valid(
            det_valid
        ),

        .det_x1(
            det_x1
        ),

        .det_y1(
            det_y1
        ),

        .det_x2(
            det_x2
        ),

        .det_y2(
            det_y2
        ),

        .det_cls(
            det_cls
        ),

        .det_score(
            det_score
        ),

        .det_anchor_idx(
            det_anchor_idx
        ),

        .n_det(
            n_det
        ),

        .overflow(
            overflow
        ),

        .done(
            done
        )
    );


    // ============================================================
    // Output activity accumulator
    //
    // This creates an observable dependency from the DUT outputs
    // to a physical top-level pin.
    //
    // Therefore synthesis cannot legally remove the complete
    // post-processing datapath.
    // ============================================================

    logic [31:0]
        result_signature;


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            result_signature <=
                32'h0000_0001;


            activity_out <=
                1'b0;

        end

        else begin

            if (det_valid) begin

                result_signature <=
                    result_signature
                    ^
                    {
                        det_cls,
                        det_score,
                        det_anchor_idx,
                        7'b0
                    }
                    ^
                    {
                        6'b0,
                        det_x1,
                        det_y1
                    }
                    ^
                    {
                        6'b0,
                        det_x2,
                        det_y2
                    };

            end


            if (done) begin

                result_signature <=
                    result_signature
                    ^
                    {
                        16'b0,
                        n_det,
                        overflow,
                        7'b0
                    };

            end


            // ----------------------------------------------------
            // Physical observable output.
            // ----------------------------------------------------

            activity_out <=
                ^result_signature;

        end

    end


endmodule