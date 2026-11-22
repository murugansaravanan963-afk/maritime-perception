`timescale 1ns / 1ps

module yolov8_postprocess_top #(
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

    parameter integer ANCHOR_BITS = 15,
    parameter integer N_ANCHORS   = 18900,

    parameter integer MAX_CAND = 4096,
    parameter integer MAX_DET  = 100,

    parameter integer IOU_NUM = 9,
    parameter integer IOU_DEN = 20,

    parameter integer AREA_BITS = 28
)(
    input logic clk,
    input logic rst_n,

    input logic frame_start,

    input logic in_valid,

    output logic in_ready,

    input logic [ANCHOR_BITS-1:0]
        anchor_idx,

    input logic [Q_BITS-1:0]
        bbox_q [0:DFL_SIDES*DFL_BINS-1],

    input logic [SCORE_BITS-1:0]
        score_q [0:NUM_CLASSES-1],

    output logic det_valid,

    output logic [COORD_BITS-1:0] det_x1,
    output logic [COORD_BITS-1:0] det_y1,
    output logic [COORD_BITS-1:0] det_x2,
    output logic [COORD_BITS-1:0] det_y2,

    output logic [CLASS_BITS-1:0]
        det_cls,

    output logic [SCORE_BITS-1:0]
        det_score,

    output logic [ANCHOR_BITS-1:0]
        det_anchor_idx,

    output logic [
        $clog2(MAX_DET+1)-1:0
    ] n_det,

    output logic overflow,

    output logic done
);


    // ============================================================
    // Raw input handshake
    // ============================================================

    wire accept_input =
        in_valid &&
        in_ready;


    // ============================================================
    // Decode interface
    // ============================================================

    logic decode_valid;
    logic decode_in_ready;


    logic [COORD_BITS-1:0]
        decode_x1;

    logic [COORD_BITS-1:0]
        decode_y1;

    logic [COORD_BITS-1:0]
        decode_x2;

    logic [COORD_BITS-1:0]
        decode_y2;


    logic [CLASS_BITS-1:0]
        decode_cls;


    logic [SCORE_BITS-1:0]
        decode_score;


    logic
        decode_pass;


    logic [ANCHOR_BITS-1:0]
        decode_anchor_idx;


    // ============================================================
    // Top-level ready
    // ============================================================

    always_comb begin

        in_ready =
            decode_in_ready;

    end


    // ============================================================
    // V5 Decode
    //
    // Supports multiple outstanding anchors.
    //
    // Anchor ID is carried through the decode pipeline itself.
    // ============================================================

    yolov8_decode_top #(
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
        ),

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
        ),

        .IMG_W(
            IMG_W
        ),

        .IMG_H(
            IMG_H
        ),

        .COORD_BITS(
            COORD_BITS
        ),

        .COORD_FRAC(
            COORD_FRAC
        ),

        .ANCHOR_BITS(
            ANCHOR_BITS
        )
    ) u_decode (
        .clk(
            clk
        ),

        .rst_n(
            rst_n
        ),

        .in_valid(
            in_valid
        ),

        .in_ready(
            decode_in_ready
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

        .out_valid(
            decode_valid
        ),

        .x1(
            decode_x1
        ),

        .y1(
            decode_y1
        ),

        .x2(
            decode_x2
        ),

        .y2(
            decode_y2
        ),

        .cls(
            decode_cls
        ),

        .score(
            decode_score
        ),

        .pass(
            decode_pass
        ),

        .out_anchor_idx(
            decode_anchor_idx
        )
    );


    // ============================================================
    // Decode-side frame end
    //
    // The frame ends only when the LAST anchor has completed the
    // complete Decode pipeline.
    //
    // This remains correct even with multiple outstanding anchors
    // because decode_anchor_idx travels with decode_valid.
    // ============================================================

    logic decode_frame_end;


    always_comb begin

        decode_frame_end =
            decode_valid
            &&
            (
                decode_anchor_idx ==
                N_ANCHORS - 1
            );

    end


    // ============================================================
    // NMS
    // ============================================================

    yolov8_nms_top #(
        .COORD_BITS(
            COORD_BITS
        ),

        .NUM_CLASSES(
            NUM_CLASSES
        ),

        .CLASS_BITS(
            CLASS_BITS
        ),

        .SCORE_BITS(
            SCORE_BITS
        ),

        .ANCHOR_BITS(
            ANCHOR_BITS
        ),

        .MAX_CAND(
            MAX_CAND
        ),

        .MAX_DET(
            MAX_DET
        ),

        .IOU_NUM(
            IOU_NUM
        ),

        .IOU_DEN(
            IOU_DEN
        ),

        .AREA_BITS(
            AREA_BITS
        )
    ) u_nms (
        .clk(
            clk
        ),

        .rst_n(
            rst_n
        ),

        .frame_start(
            frame_start
        ),

        .frame_end(
            decode_frame_end
        ),

        .in_valid(
            decode_valid
        ),

        .in_x1(
            decode_x1
        ),

        .in_y1(
            decode_y1
        ),

        .in_x2(
            decode_x2
        ),

        .in_y2(
            decode_y2
        ),

        .in_cls(
            decode_cls
        ),

        .in_score(
            decode_score
        ),

        .in_pass(
            decode_pass
        ),

        .in_anchor_idx(
            decode_anchor_idx
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
    // Simulation checks
    // ============================================================

`ifndef SYNTHESIS

    initial begin

        if (
            DFL_BINS != 16
        ) begin

            $error(
                "yolov8_postprocess_top V5 requires DFL_BINS=16"
            );

        end


        if (
            DFL_SIDES != 4
        ) begin

            $error(
                "yolov8_postprocess_top V5 requires DFL_SIDES=4"
            );

        end


        if (
            N_ANCHORS < 1
        ) begin

            $error(
                "yolov8_postprocess_top: N_ANCHORS must be >= 1"
            );

        end


        if (
            (1 << ANCHOR_BITS) <
            N_ANCHORS
        ) begin

            $error(
                "yolov8_postprocess_top: ANCHOR_BITS too small"
            );

        end


        if (
            MAX_CAND < 1
        ) begin

            $error(
                "yolov8_postprocess_top: MAX_CAND must be >= 1"
            );

        end


        if (
            MAX_DET < 1
        ) begin

            $error(
                "yolov8_postprocess_top: MAX_DET must be >= 1"
            );

        end

    end


    // ============================================================
    // Runtime checks
    //
    // NOTE:
    // The old "multiple outstanding anchors are not supported"
    // assertion is intentionally removed.
    //
    // V5 explicitly supports multiple outstanding anchors.
    // ============================================================

    always_ff @(posedge clk) begin

        if (rst_n) begin

            if (
                accept_input
                &&
                anchor_idx >=
                N_ANCHORS
            ) begin

                $error(
                    "yolov8_postprocess_top: accepted anchor index out of range"
                );

            end


            if (
                decode_valid
                &&
                decode_anchor_idx >=
                N_ANCHORS
            ) begin

                $error(
                    "yolov8_postprocess_top: decoded anchor index out of range"
                );

            end

        end

    end

`endif


endmodule