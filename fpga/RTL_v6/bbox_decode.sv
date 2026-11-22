`timescale 1ns / 1ps

module bbox_decode #(
    parameter integer IMG_W = 960,
    parameter integer IMG_H = 960,

    // ============================================================
    // YOLOv8 feature-map geometry
    // ============================================================

    parameter integer GRID8_W  = 120,
    parameter integer GRID8_H  = 120,

    parameter integer GRID16_W = 60,
    parameter integer GRID16_H = 60,

    parameter integer GRID32_W = 30,
    parameter integer GRID32_H = 30,

    // ============================================================
    // Widths
    // ============================================================

    parameter integer COORD_BITS  = 14,
    parameter integer ANCHOR_BITS = 15,

    parameter integer OFFSET_BITS = 12,

    parameter integer OFFSET_FRAC = 8,
    parameter integer COORD_FRAC  = 4
)(
    input logic clk,
    input logic rst_n,

    input logic in_valid,

    input logic [ANCHOR_BITS-1:0]
        anchor_idx,

    input logic [OFFSET_BITS-1:0] off_l,
    input logic [OFFSET_BITS-1:0] off_t,
    input logic [OFFSET_BITS-1:0] off_r,
    input logic [OFFSET_BITS-1:0] off_b,

    output logic out_valid,

    output logic [COORD_BITS-1:0] x1,
    output logic [COORD_BITS-1:0] y1,
    output logic [COORD_BITS-1:0] x2,
    output logic [COORD_BITS-1:0] y2
);


    // ============================================================
    // Anchor ranges
    // ============================================================

    localparam integer N_S8 =
        GRID8_W * GRID8_H;

    localparam integer N_S16 =
        GRID16_W * GRID16_H;

    localparam integer N_S32 =
        GRID32_W * GRID32_H;

    localparam integer START_S16 =
        N_S8;

    localparam integer START_S32 =
        N_S8 + N_S16;

    localparam integer TOTAL_ANCHORS =
        N_S8 + N_S16 + N_S32;


    // ============================================================
    // Clipping limits
    // ============================================================

    localparam integer MAX_X_Q =
        IMG_W << COORD_FRAC;

    localparam integer MAX_Y_Q =
        IMG_H << COORD_FRAC;


    // ============================================================
    // Derived widths
    // ============================================================

    localparam integer MAX_GRID_W =
        (GRID8_W > GRID16_W) ?
            ((GRID8_W > GRID32_W) ?
                GRID8_W : GRID32_W)
            :
            ((GRID16_W > GRID32_W) ?
                GRID16_W : GRID32_W);


    localparam integer MAX_GRID_H =
        (GRID8_H > GRID16_H) ?
            ((GRID8_H > GRID32_H) ?
                GRID8_H : GRID32_H)
            :
            ((GRID16_H > GRID32_H) ?
                GRID16_H : GRID32_H);


    localparam integer GRID_X_BITS =
        (MAX_GRID_W <= 1) ?
            1 : $clog2(MAX_GRID_W);


    localparam integer GRID_Y_BITS =
        (MAX_GRID_H <= 1) ?
            1 : $clog2(MAX_GRID_H);


    localparam integer STRIDE_LOG2_BITS =
        3;


    localparam integer MAX_STRIDE_LOG2 =
        5;


    localparam integer OFFSET_EXT_BITS =
        OFFSET_BITS +
        MAX_STRIDE_LOG2;


    localparam integer CALC_BITS =
        (
            (COORD_BITS + 2) >
            (OFFSET_EXT_BITS + 1)
        )
        ?
            (COORD_BITS + 2)
        :
            (OFFSET_EXT_BITS + 1);


    // ============================================================
    // Stage 1: streaming grid generator
    //
    // The complete YOLOv8 post-processing path presents anchors in
    // deterministic order:
    //
    //     0, 1, 2, ... , TOTAL_ANCHORS-1
    //
    // Therefore grid_x/grid_y do not need to be recomputed from
    // anchor_idx using division/modulo or repeated subtraction.
    // Instead, small counters advance once for every accepted bbox.
    //
    // This changes grid generation from an anchor-dependent
    // 1..52-cycle loop into a fixed one-cycle streaming stage.
    // ============================================================

    logic [GRID_X_BITS-1:0]
        grid_x_counter;

    logic [GRID_Y_BITS-1:0]
        grid_y_counter;

    logic [ANCHOR_BITS-1:0]
        expected_anchor_idx;


    // ============================================================
    // Stage-1 result registers
    // ============================================================

    logic [GRID_X_BITS-1:0]
        grid_x_s1;

    logic [GRID_Y_BITS-1:0]
        grid_y_s1;

    logic [STRIDE_LOG2_BITS-1:0]
        stride_log2_s1;

    logic [OFFSET_BITS-1:0]
        off_l_s1;

    logic [OFFSET_BITS-1:0]
        off_t_s1;

    logic [OFFSET_BITS-1:0]
        off_r_s1;

    logic [OFFSET_BITS-1:0]
        off_b_s1;

    logic valid_s1;


    // ============================================================
    // Streaming grid counter
    // ============================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            grid_x_counter <=
                '0;

            grid_y_counter <=
                '0;

            expected_anchor_idx <=
                '0;

            grid_x_s1 <=
                '0;

            grid_y_s1 <=
                '0;

            stride_log2_s1 <=
                '0;

            off_l_s1 <=
                '0;

            off_t_s1 <=
                '0;

            off_r_s1 <=
                '0;

            off_b_s1 <=
                '0;

            valid_s1 <=
                1'b0;

        end

        else begin

            // ----------------------------------------------------
            // One-cycle pulse.
            // ----------------------------------------------------

            valid_s1 <=
                1'b0;


            if (in_valid) begin

                // ------------------------------------------------
                // Current anchor uses the current grid counters.
                // ------------------------------------------------

                grid_x_s1 <=
                    grid_x_counter;

                grid_y_s1 <=
                    grid_y_counter;


                if (
                    anchor_idx <
                    START_S16
                ) begin

                    stride_log2_s1 <=
                        3'd3;

                end

                else if (
                    anchor_idx <
                    START_S32
                ) begin

                    stride_log2_s1 <=
                        3'd4;

                end

                else begin

                    stride_log2_s1 <=
                        3'd5;

                end


                // ------------------------------------------------
                // Capture DFL offsets.
                // ------------------------------------------------

                off_l_s1 <=
                    off_l;

                off_t_s1 <=
                    off_t;

                off_r_s1 <=
                    off_r;

                off_b_s1 <=
                    off_b;

                valid_s1 <=
                    1'b1;


                // ------------------------------------------------
                // Advance expected anchor ID.
                // ------------------------------------------------

                if (
                    anchor_idx ==
                    TOTAL_ANCHORS - 1
                ) begin

                    expected_anchor_idx <=
                        '0;

                end
                else begin

                    expected_anchor_idx <=
                        expected_anchor_idx + 1'b1;

                end


                // ------------------------------------------------
                // Advance grid counters.
                //
                // At the end of each feature-map scale, reset both
                // counters so the next accepted anchor starts at
                // (0,0) of the next scale.
                // ------------------------------------------------

                if (
                    anchor_idx ==
                    START_S16 - 1
                ) begin

                    grid_x_counter <=
                        '0;

                    grid_y_counter <=
                        '0;

                end

                else if (
                    anchor_idx ==
                    START_S32 - 1
                ) begin

                    grid_x_counter <=
                        '0;

                    grid_y_counter <=
                        '0;

                end

                else if (
                    anchor_idx ==
                    TOTAL_ANCHORS - 1
                ) begin

                    grid_x_counter <=
                        '0;

                    grid_y_counter <=
                        '0;

                end

                // ------------------------------------------------
                // Stride-8 grid: GRID8_W columns (120 for Maritime6-960).
                // ------------------------------------------------

                else if (
                    anchor_idx <
                    START_S16
                ) begin

                    if (
                        grid_x_counter ==
                        GRID8_W - 1
                    ) begin

                        grid_x_counter <=
                            '0;

                        grid_y_counter <=
                            grid_y_counter + 1'b1;

                    end
                    else begin

                        grid_x_counter <=
                            grid_x_counter + 1'b1;

                    end

                end

                // ------------------------------------------------
                // Stride-16 grid: GRID16_W columns (60 for Maritime6-960).
                // ------------------------------------------------

                else if (
                    anchor_idx <
                    START_S32
                ) begin

                    if (
                        grid_x_counter ==
                        GRID16_W - 1
                    ) begin

                        grid_x_counter <=
                            '0;

                        grid_y_counter <=
                            grid_y_counter + 1'b1;

                    end
                    else begin

                        grid_x_counter <=
                            grid_x_counter + 1'b1;

                    end

                end

                // ------------------------------------------------
                // Stride-32 grid: GRID32_W columns (30 for Maritime6-960).
                // ------------------------------------------------

                else begin

                    if (
                        grid_x_counter ==
                        GRID32_W - 1
                    ) begin

                        grid_x_counter <=
                            '0;

                        grid_y_counter <=
                            grid_y_counter + 1'b1;

                    end
                    else begin

                        grid_x_counter <=
                            grid_x_counter + 1'b1;

                    end

                end

            end

        end

    end


    // ============================================================
    // Stage 2
    //
    // Anchor center and DFL pixel-distance conversion.
    // ============================================================

    logic signed [CALC_BITS-1:0]
        cx_s2;

    logic signed [CALC_BITS-1:0]
        cy_s2;


    logic signed [CALC_BITS-1:0]
        off_l_px_s2;

    logic signed [CALC_BITS-1:0]
        off_t_px_s2;

    logic signed [CALC_BITS-1:0]
        off_r_px_s2;

    logic signed [CALC_BITS-1:0]
        off_b_px_s2;


    logic valid_s2;


    // ============================================================
    // Explicitly widened temporary values
    // ============================================================

    logic [CALC_BITS-1:0]
        grid_x_ext;

    logic [CALC_BITS-1:0]
        grid_y_ext;


    logic [CALC_BITS-1:0]
        off_l_ext;

    logic [CALC_BITS-1:0]
        off_t_ext;

    logic [CALC_BITS-1:0]
        off_r_ext;

    logic [CALC_BITS-1:0]
        off_b_ext;


    always_comb begin

        grid_x_ext =
            '0;

        grid_y_ext =
            '0;


        off_l_ext =
            '0;

        off_t_ext =
            '0;

        off_r_ext =
            '0;

        off_b_ext =
            '0;


        grid_x_ext[
            GRID_X_BITS-1:0
        ] =
            grid_x_s1;


        grid_y_ext[
            GRID_Y_BITS-1:0
        ] =
            grid_y_s1;


        off_l_ext[
            OFFSET_BITS-1:0
        ] =
            off_l_s1;


        off_t_ext[
            OFFSET_BITS-1:0
        ] =
            off_t_s1;


        off_r_ext[
            OFFSET_BITS-1:0
        ] =
            off_r_s1;


        off_b_ext[
            OFFSET_BITS-1:0
        ] =
            off_b_s1;

    end


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            cx_s2 <=
                '0;

            cy_s2 <=
                '0;


            off_l_px_s2 <=
                '0;

            off_t_px_s2 <=
                '0;

            off_r_px_s2 <=
                '0;

            off_b_px_s2 <=
                '0;


            valid_s2 <=
                1'b0;

        end

        else begin

            valid_s2 <=
                valid_s1;


            if (valid_s1) begin

                // ------------------------------------------------
                // Anchor center
                // ------------------------------------------------

                cx_s2 <=
                    $signed(
                        (
                            (
                                (grid_x_ext << 1)
                                +
                                1'b1
                            )
                            <<
                            stride_log2_s1
                        )
                        <<
                        (COORD_FRAC - 1)
                    );


                cy_s2 <=
                    $signed(
                        (
                            (
                                (grid_y_ext << 1)
                                +
                                1'b1
                            )
                            <<
                            stride_log2_s1
                        )
                        <<
                        (COORD_FRAC - 1)
                    );


                // ------------------------------------------------
                // DFL offsets -> pixel distance
                // ------------------------------------------------

                off_l_px_s2 <=
                    $signed(
                        (
                            off_l_ext
                            <<
                            stride_log2_s1
                        )
                        >>
                        (
                            OFFSET_FRAC -
                            COORD_FRAC
                        )
                    );


                off_t_px_s2 <=
                    $signed(
                        (
                            off_t_ext
                            <<
                            stride_log2_s1
                        )
                        >>
                        (
                            OFFSET_FRAC -
                            COORD_FRAC
                        )
                    );


                off_r_px_s2 <=
                    $signed(
                        (
                            off_r_ext
                            <<
                            stride_log2_s1
                        )
                        >>
                        (
                            OFFSET_FRAC -
                            COORD_FRAC
                        )
                    );


                off_b_px_s2 <=
                    $signed(
                        (
                            off_b_ext
                            <<
                            stride_log2_s1
                        )
                        >>
                        (
                            OFFSET_FRAC -
                            COORD_FRAC
                        )
                    );

            end

        end

    end


    // ============================================================
    // Stage 3
    //
    // Un-clipped box:
    //
    // x1 = cx - left
    // y1 = cy - top
    // x2 = cx + right
    // y2 = cy + bottom
    // ============================================================

    logic signed [CALC_BITS-1:0]
        raw_x1_s3;

    logic signed [CALC_BITS-1:0]
        raw_y1_s3;

    logic signed [CALC_BITS-1:0]
        raw_x2_s3;

    logic signed [CALC_BITS-1:0]
        raw_y2_s3;


    logic valid_s3;


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            raw_x1_s3 <=
                '0;

            raw_y1_s3 <=
                '0;

            raw_x2_s3 <=
                '0;

            raw_y2_s3 <=
                '0;


            valid_s3 <=
                1'b0;

        end

        else begin

            valid_s3 <=
                valid_s2;


            if (valid_s2) begin

                raw_x1_s3 <=
                    cx_s2 -
                    off_l_px_s2;


                raw_y1_s3 <=
                    cy_s2 -
                    off_t_px_s2;


                raw_x2_s3 <=
                    cx_s2 +
                    off_r_px_s2;


                raw_y2_s3 <=
                    cy_s2 +
                    off_b_px_s2;

            end

        end

    end


    // ============================================================
    // Stage 4
    //
    // Clip final coordinates.
    // ============================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            x1 <=
                '0;

            y1 <=
                '0;

            x2 <=
                '0;

            y2 <=
                '0;


            out_valid <=
                1'b0;

        end

        else begin

            out_valid <=
                valid_s3;


            if (valid_s3) begin

                // ------------------------------------------------
                // X1
                // ------------------------------------------------

                if (
                    raw_x1_s3 <
                    0
                ) begin

                    x1 <=
                        '0;

                end

                else if (
                    raw_x1_s3 >
                    MAX_X_Q
                ) begin

                    x1 <=
                        MAX_X_Q[
                            COORD_BITS-1:0
                        ];

                end

                else begin

                    x1 <=
                        raw_x1_s3[
                            COORD_BITS-1:0
                        ];

                end


                // ------------------------------------------------
                // Y1
                // ------------------------------------------------

                if (
                    raw_y1_s3 <
                    0
                ) begin

                    y1 <=
                        '0;

                end

                else if (
                    raw_y1_s3 >
                    MAX_Y_Q
                ) begin

                    y1 <=
                        MAX_Y_Q[
                            COORD_BITS-1:0
                        ];

                end

                else begin

                    y1 <=
                        raw_y1_s3[
                            COORD_BITS-1:0
                        ];

                end


                // ------------------------------------------------
                // X2
                // ------------------------------------------------

                if (
                    raw_x2_s3 <
                    0
                ) begin

                    x2 <=
                        '0;

                end

                else if (
                    raw_x2_s3 >
                    MAX_X_Q
                ) begin

                    x2 <=
                        MAX_X_Q[
                            COORD_BITS-1:0
                        ];

                end

                else begin

                    x2 <=
                        raw_x2_s3[
                            COORD_BITS-1:0
                        ];

                end


                // ------------------------------------------------
                // Y2
                // ------------------------------------------------

                if (
                    raw_y2_s3 <
                    0
                ) begin

                    y2 <=
                        '0;

                end

                else if (
                    raw_y2_s3 >
                    MAX_Y_Q
                ) begin

                    y2 <=
                        MAX_Y_Q[
                            COORD_BITS-1:0
                        ];

                end

                else begin

                    y2 <=
                        raw_y2_s3[
                            COORD_BITS-1:0
                        ];

                end

            end

        end

    end


    // ============================================================
    // Simulation-only checks
    // ============================================================

`ifndef SYNTHESIS

    initial begin

        if (
            OFFSET_FRAC <
            COORD_FRAC
        ) begin

            $error(
                "bbox_decode: OFFSET_FRAC must be >= COORD_FRAC"
            );

        end


        if (
            (1 << ANCHOR_BITS) <
            TOTAL_ANCHORS
        ) begin

            $error(
                "bbox_decode: ANCHOR_BITS is too small for TOTAL_ANCHORS"
            );

        end

    end


    // ============================================================
    // Streaming-grid protocol check.
    //
    // The counter-based implementation assumes the complete decode
    // stream presents anchors in deterministic ascending order.
    // ============================================================

    always_ff @(posedge clk) begin

        if (
            rst_n &&
            in_valid &&
            (anchor_idx != expected_anchor_idx)
        ) begin

            $error(
                "bbox_decode: streaming grid expected anchor %0d but received %0d",
                expected_anchor_idx,
                anchor_idx
            );

        end

    end

`endif

endmodule