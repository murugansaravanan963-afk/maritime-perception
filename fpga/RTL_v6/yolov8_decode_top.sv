`timescale 1ns / 1ps

module yolov8_decode_top #(
    parameter integer Q_BITS = 8,

    parameter integer DFL_BINS  = 16,
    parameter integer DFL_SIDES = 4,

    parameter integer LUT_WIDTH = 16,
    parameter integer LUT_DEPTH = 65,

    parameter LUT_FILE =
        "dfl_exp_lut.mem",

    parameter integer OFF_BITS = 12,
    parameter integer OFF_FRAC = 8,

    parameter integer NUM_CLASSES = 6,
    parameter integer SCORE_BITS  = 8,

    parameter integer CLASS_BITS =
        (NUM_CLASSES <= 1)
        ? 1
        : $clog2(NUM_CLASSES),

    parameter integer SCORE_Q_THR = 64,

    parameter integer IMG_W = 960,
    parameter integer IMG_H = 960,

    parameter integer COORD_BITS = 14,
    parameter integer COORD_FRAC = 4,

    parameter integer ANCHOR_BITS = 15
)(
    input logic clk,
    input logic rst_n,

    input logic in_valid,
    output logic in_ready,

    input logic [ANCHOR_BITS-1:0]
        anchor_idx,

    input logic [Q_BITS-1:0]
        bbox_q [0:DFL_SIDES*DFL_BINS-1],

    input logic [SCORE_BITS-1:0]
        score_q [0:NUM_CLASSES-1],

    output logic out_valid,

    output logic [COORD_BITS-1:0] x1,
    output logic [COORD_BITS-1:0] y1,
    output logic [COORD_BITS-1:0] x2,
    output logic [COORD_BITS-1:0] y2,

    output logic [CLASS_BITS-1:0] cls,

    output logic [SCORE_BITS-1:0]
        score,

    output logic pass,

    // ============================================================
    // V5:
    // Anchor ID is now part of the decoded transaction.
    // ============================================================

    output logic [ANCHOR_BITS-1:0]
        out_anchor_idx
);


    // ============================================================
    // V5 architecture
    //
    // Raw input
    //    |
    //    +-------------------+
    //    |                   |
    //    v                   v
    // Dual-bank DFL       Score pipeline
    //    |                   |
    //    |             Metadata FIFO
    //    |                   |
    //    +--------+----------+
    //             |
    //        ordered retire
    //             |
    //         bbox_launch
    //             |
    //        bbox_decode
    //             |
    //       final decoded
    //
    // IMPORTANT:
    //
    // Metadata FIFO is popped at bbox_launch, NOT at out_valid.
    //
    // This allows the next DFL result to use the next metadata
    // entry even while the previous result is still travelling
    // through the four-stage bbox pipeline.
    // ============================================================


    // ============================================================
    // Metadata FIFO
    // ============================================================

    localparam integer META_DEPTH =
        4;

    localparam integer META_ADDR_BITS =
        2;

    localparam integer META_COUNT_BITS =
        3;


    logic [ANCHOR_BITS-1:0]
        meta_anchor [0:META_DEPTH-1];

    logic [CLASS_BITS-1:0]
        meta_cls [0:META_DEPTH-1];

    logic [SCORE_BITS-1:0]
        meta_score [0:META_DEPTH-1];

    logic
        meta_pass [0:META_DEPTH-1];

    logic
        meta_score_ready [0:META_DEPTH-1];


    logic [META_ADDR_BITS-1:0]
        meta_wr_ptr;

    logic [META_ADDR_BITS-1:0]
        meta_rd_ptr;

    logic [META_ADDR_BITS-1:0]
        score_wr_ptr;

    logic [META_COUNT_BITS-1:0]
        meta_count;


    wire meta_full =
        (
            meta_count ==
            META_DEPTH
        );


    wire meta_empty =
        (
            meta_count ==
            0
        );


    // ============================================================
    // DFL
    // ============================================================

    logic dfl_in_ready;
    logic dfl_valid;


    logic [OFF_BITS-1:0]
        off_l;

    logic [OFF_BITS-1:0]
        off_t;

    logic [OFF_BITS-1:0]
        off_r;

    logic [OFF_BITS-1:0]
        off_b;


    // ============================================================
    // Input handshake
    // ============================================================

    always_comb begin

        in_ready =
            dfl_in_ready &&
            !meta_full;

    end


    wire accept_input =
        in_valid &&
        in_ready;


    // ============================================================
    // V5 dual-bank DFL
    // ============================================================

    yolov8_dfl_top #(
        .Q_BITS(
            Q_BITS
        ),

        .DFL_BINS(
            DFL_BINS
        ),

        .DFL_SIDES(
            DFL_SIDES
        ),

        .LUT_WIDTH(
            LUT_WIDTH
        ),

        .LUT_DEPTH(
            LUT_DEPTH
        ),

        .LUT_FILE(
            LUT_FILE
        ),

        .OFF_BITS(
            OFF_BITS
        ),

        .OFF_FRAC(
            OFF_FRAC
        )
    ) u_dfl (
        .clk(
            clk
        ),

        .rst_n(
            rst_n
        ),

        .in_valid(
            accept_input
        ),

        .in_ready(
            dfl_in_ready
        ),

        .bbox_q(
            bbox_q
        ),

        .out_valid(
            dfl_valid
        ),

        .off_l(
            off_l
        ),

        .off_t(
            off_t
        ),

        .off_r(
            off_r
        ),

        .off_b(
            off_b
        )
    );


    // ============================================================
    // Score path
    // ============================================================

    logic score_valid;

    logic [CLASS_BITS-1:0]
        score_cls;

    logic [SCORE_BITS-1:0]
        score_max;

    logic
        score_pass;


    score_argmax_threshold #(
        .NUM_CLASSES(
            NUM_CLASSES
        ),

        .SCORE_BITS(
            SCORE_BITS
        ),

        .CLASS_BITS(
            CLASS_BITS
        ),

        .SCORE_Q_THR(
            SCORE_Q_THR
        )
    ) u_score (
        .clk(
            clk
        ),

        .rst_n(
            rst_n
        ),

        .in_valid(
            accept_input
        ),

        .score_q(
            score_q
        ),

        .out_valid(
            score_valid
        ),

        .cls(
            score_cls
        ),

        .score(
            score_max
        ),

        .pass(
            score_pass
        )
    );


    // ============================================================
    // FIFO head
    // ============================================================

    wire [ANCHOR_BITS-1:0]
        head_anchor =
            meta_anchor[
                meta_rd_ptr
            ];


    wire [CLASS_BITS-1:0]
        head_cls =
            meta_cls[
                meta_rd_ptr
            ];


    wire [SCORE_BITS-1:0]
        head_score =
            meta_score[
                meta_rd_ptr
            ];


    wire
        head_pass =
            meta_pass[
                meta_rd_ptr
            ];


    wire
        head_score_ready =
            meta_score_ready[
                meta_rd_ptr
            ];


    // ============================================================
    // BBox launch
    //
    // DFL output is already ordered.
    //
    // The oldest metadata entry therefore belongs to the current
    // DFL result.
    //
    // Pop happens HERE.
    // ============================================================

    wire bbox_launch =
        dfl_valid &&
        !meta_empty &&
        head_score_ready;


    // ============================================================
    // Metadata FIFO control
    // ============================================================

    integer meta_i;


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            meta_wr_ptr <=
                '0;

            meta_rd_ptr <=
                '0;

            score_wr_ptr <=
                '0;

            meta_count <=
                '0;


            for (
                meta_i = 0;
                meta_i < META_DEPTH;
                meta_i = meta_i + 1
            ) begin

                meta_anchor[meta_i] <=
                    '0;

                meta_cls[meta_i] <=
                    '0;

                meta_score[meta_i] <=
                    '0;

                meta_pass[meta_i] <=
                    1'b0;

                meta_score_ready[meta_i] <=
                    1'b0;

            end

        end

        else begin

            // ====================================================
            // New transaction
            // ====================================================

            if (accept_input) begin

                meta_anchor[
                    meta_wr_ptr
                ] <=
                    anchor_idx;


                meta_score_ready[
                    meta_wr_ptr
                ] <=
                    1'b0;


                meta_wr_ptr <=
                    meta_wr_ptr +
                    1'b1;

            end


            // ====================================================
            // Score result
            //
            // Score pipeline preserves input order.
            // ====================================================

            if (score_valid) begin

                meta_cls[
                    score_wr_ptr
                ] <=
                    score_cls;


                meta_score[
                    score_wr_ptr
                ] <=
                    score_max;


                meta_pass[
                    score_wr_ptr
                ] <=
                    score_pass;


                meta_score_ready[
                    score_wr_ptr
                ] <=
                    1'b1;


                score_wr_ptr <=
                    score_wr_ptr +
                    1'b1;

            end


            // ====================================================
            // Pop metadata when DFL result ENTERS bbox pipeline.
            // ====================================================

            if (bbox_launch) begin

                meta_score_ready[
                    meta_rd_ptr
                ] <=
                    1'b0;


                meta_rd_ptr <=
                    meta_rd_ptr +
                    1'b1;

            end


            // ====================================================
            // FIFO count
            // ====================================================

            case ({
                accept_input,
                bbox_launch
            })

                2'b10:
                begin

                    meta_count <=
                        meta_count +
                        1'b1;

                end


                2'b01:
                begin

                    meta_count <=
                        meta_count -
                        1'b1;

                end


                2'b11:
                begin

                    meta_count <=
                        meta_count;

                end


                default:
                begin

                    meta_count <=
                        meta_count;

                end

            endcase

        end

    end


    // ============================================================
    // BBox pipeline
    // ============================================================

    logic bbox_valid;


    logic [COORD_BITS-1:0]
        bbox_x1;

    logic [COORD_BITS-1:0]
        bbox_y1;

    logic [COORD_BITS-1:0]
        bbox_x2;

    logic [COORD_BITS-1:0]
        bbox_y2;


    bbox_decode #(
        .IMG_W(
            IMG_W
        ),

        .IMG_H(
            IMG_H
        ),

        .GRID8_W(
            IMG_W / 8
        ),

        .GRID8_H(
            IMG_H / 8
        ),

        .GRID16_W(
            IMG_W / 16
        ),

        .GRID16_H(
            IMG_H / 16
        ),

        .GRID32_W(
            IMG_W / 32
        ),

        .GRID32_H(
            IMG_H / 32
        ),

        .COORD_BITS(
            COORD_BITS
        ),

        .ANCHOR_BITS(
            ANCHOR_BITS
        ),

        .OFFSET_BITS(
            OFF_BITS
        ),

        .OFFSET_FRAC(
            OFF_FRAC
        ),

        .COORD_FRAC(
            COORD_FRAC
        )
    ) u_bbox_decode (
        .clk(
            clk
        ),

        .rst_n(
            rst_n
        ),

        .in_valid(
            bbox_launch
        ),

        .anchor_idx(
            head_anchor
        ),

        .off_l(
            off_l
        ),

        .off_t(
            off_t
        ),

        .off_r(
            off_r
        ),

        .off_b(
            off_b
        ),

        .out_valid(
            bbox_valid
        ),

        .x1(
            bbox_x1
        ),

        .y1(
            bbox_y1
        ),

        .x2(
            bbox_x2
        ),

        .y2(
            bbox_y2
        )
    );


    // ============================================================
    // Metadata pipeline
    //
    // bbox_decode has four registered stages.
    //
    // Metadata therefore travels through four matching stages.
    //
    // This is essential because dual-bank DFL results may arrive
    // only two clocks apart.
    // ============================================================

    logic [ANCHOR_BITS-1:0]
        anchor_pipe [0:3];

    logic [CLASS_BITS-1:0]
        cls_pipe [0:3];

    logic [SCORE_BITS-1:0]
        score_pipe [0:3];

    logic
        pass_pipe [0:3];

    logic
        meta_valid_pipe [0:3];


    integer pipe_i;


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            for (
                pipe_i = 0;
                pipe_i < 4;
                pipe_i = pipe_i + 1
            ) begin

                anchor_pipe[pipe_i] <=
                    '0;

                cls_pipe[pipe_i] <=
                    '0;

                score_pipe[pipe_i] <=
                    '0;

                pass_pipe[pipe_i] <=
                    1'b0;

                meta_valid_pipe[pipe_i] <=
                    1'b0;

            end

        end

        else begin

            // ====================================================
            // Stage 0
            // ====================================================

            meta_valid_pipe[0] <=
                bbox_launch;


            if (bbox_launch) begin

                anchor_pipe[0] <=
                    head_anchor;

                cls_pipe[0] <=
                    head_cls;

                score_pipe[0] <=
                    head_score;

                pass_pipe[0] <=
                    head_pass;

            end


            // ====================================================
            // Stages 1..3
            // ====================================================

            for (
                pipe_i = 1;
                pipe_i < 4;
                pipe_i = pipe_i + 1
            ) begin

                meta_valid_pipe[pipe_i] <=
                    meta_valid_pipe[
                        pipe_i - 1
                    ];


                if (
                    meta_valid_pipe[
                        pipe_i - 1
                    ]
                ) begin

                    anchor_pipe[pipe_i] <=
                        anchor_pipe[
                            pipe_i - 1
                        ];


                    cls_pipe[pipe_i] <=
                        cls_pipe[
                            pipe_i - 1
                        ];


                    score_pipe[pipe_i] <=
                        score_pipe[
                            pipe_i - 1
                        ];


                    pass_pipe[pipe_i] <=
                        pass_pipe[
                            pipe_i - 1
                        ];

                end

            end

        end

    end


    // ============================================================
    // Final output
    //
    // bbox_valid and metadata pipeline stage 3 correspond to the
    // same transaction.
    // ============================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            out_valid <=
                1'b0;


            x1 <=
                '0;

            y1 <=
                '0;

            x2 <=
                '0;

            y2 <=
                '0;


            cls <=
                '0;

            score <=
                '0;

            pass <=
                1'b0;


            out_anchor_idx <=
                '0;

        end

        else begin

            out_valid <=
                bbox_valid;


            if (bbox_valid) begin

                x1 <=
                    bbox_x1;

                y1 <=
                    bbox_y1;

                x2 <=
                    bbox_x2;

                y2 <=
                    bbox_y2;


                cls <=
                    cls_pipe[3];


                score <=
                    score_pipe[3];


                pass <=
                    pass_pipe[3];


                out_anchor_idx <=
                    anchor_pipe[3];

            end

        end

    end


    // ============================================================
    // Simulation checks
    // ============================================================

`ifndef SYNTHESIS

    initial begin

        if (
            DFL_BINS != 16
        ) begin

            $error(
                "yolov8_decode_top V5 requires DFL_BINS=16"
            );

        end


        if (
            DFL_SIDES != 4
        ) begin

            $error(
                "yolov8_decode_top V5 requires DFL_SIDES=4"
            );

        end

    end


    always_ff @(posedge clk) begin

        if (rst_n) begin

            if (
                dfl_valid &&
                meta_empty
            ) begin

                $error(
                    "yolov8_decode_top V5: DFL result without metadata"
                );

            end


            if (
                bbox_valid &&
                !meta_valid_pipe[3]
            ) begin

                $error(
                    "yolov8_decode_top V5: bbox/metadata pipeline misalignment"
                );

            end

        end

    end

`endif


endmodule