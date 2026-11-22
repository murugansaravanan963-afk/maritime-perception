`timescale 1ns / 1ps

module yolov8_nms_top #(
    // ============================================================
    // Coordinate configuration
    // ============================================================

    parameter integer COORD_BITS = 14,

    // ============================================================
    // Class / score configuration
    // ============================================================

    parameter integer NUM_CLASSES = 6,

    parameter integer CLASS_BITS =
        (NUM_CLASSES <= 1) ?
            1 : $clog2(NUM_CLASSES),

    parameter integer SCORE_BITS = 8,

    // ============================================================
    // Anchor configuration
    // ============================================================

    parameter integer ANCHOR_BITS = 15,

    // ============================================================
    // NMS capacity
    // ============================================================

    parameter integer MAX_CAND = 4096,
    parameter integer MAX_DET  = 100,

    // ============================================================
    // IoU threshold
    //
    // suppress when:
    //
    // inter > 0
    //
    // AND
    //
    // IOU_DEN * inter >= IOU_NUM * union
    //
    // Current:
    //
    // 9 / 20 = 0.45
    // ============================================================

    parameter integer IOU_NUM = 9,
    parameter integer IOU_DEN = 20,

    // ============================================================
    // Area width
    //
    // Model-dependent.
    //
    // Current generated configuration:
    //
    // COORD_BITS = 14
    // AREA_BITS  = 26
    //
    // Final value should come from rtl_params.vh.
    // ============================================================

    parameter integer AREA_BITS = 28
)(
    input logic clk,
    input logic rst_n,

    // ============================================================
    // Frame control
    //
    // frame_start:
    // pulse before first decoded anchor
    //
    // frame_end:
    // pulse together with / immediately after final decoded anchor
    // ============================================================

    input logic frame_start,
    input logic frame_end,

    // ============================================================
    // Checkpoint-2 decoded input stream
    // ============================================================

    input logic in_valid,

    input logic [COORD_BITS-1:0] in_x1,
    input logic [COORD_BITS-1:0] in_y1,
    input logic [COORD_BITS-1:0] in_x2,
    input logic [COORD_BITS-1:0] in_y2,

    input logic [CLASS_BITS-1:0] in_cls,

    input logic [SCORE_BITS-1:0] in_score,

    input logic in_pass,

    input logic [ANCHOR_BITS-1:0]
        in_anchor_idx,

    // ============================================================
    // Final detection output stream
    // ============================================================

    output logic det_valid,

    output logic [COORD_BITS-1:0] det_x1,
    output logic [COORD_BITS-1:0] det_y1,
    output logic [COORD_BITS-1:0] det_x2,
    output logic [COORD_BITS-1:0] det_y2,

    output logic [CLASS_BITS-1:0] det_cls,

    output logic [SCORE_BITS-1:0] det_score,

    output logic [ANCHOR_BITS-1:0]
        det_anchor_idx,

    // ============================================================
    // Frame result
    // ============================================================

    output logic [
        $clog2(MAX_DET+1)-1:0
    ] n_det,

    output logic overflow,

    output logic done
);

    // ============================================================
    // Derived widths
    // ============================================================

    localparam integer CAND_ADDR_BITS =
        (MAX_CAND <= 1) ?
            1 :
            $clog2(MAX_CAND);

    localparam integer CAND_COUNT_BITS =
        $clog2(MAX_CAND + 1);

    localparam integer DET_COUNT_BITS =
        $clog2(MAX_DET + 1);


    // ============================================================
    // Candidate Collector <-> NMS Engine
    // ============================================================

    logic [CAND_COUNT_BITS-1:0]
        cand_count;

    logic collect_done;


    // ============================================================
    // Candidate RAM read address
    // ============================================================

    logic [CAND_ADDR_BITS-1:0]
        cand_rd_addr;


    // ============================================================
    // Candidate RAM read data
    // ============================================================

    logic [COORD_BITS-1:0] cand_x1;
    logic [COORD_BITS-1:0] cand_y1;
    logic [COORD_BITS-1:0] cand_x2;
    logic [COORD_BITS-1:0] cand_y2;

    logic [CLASS_BITS-1:0]
        cand_cls;

    logic [SCORE_BITS-1:0]
        cand_score;

    logic [ANCHOR_BITS-1:0]
        cand_anchor;


    // ============================================================
    // Candidate Collector
    //
    // Golden behavior:
    //
    // cand = nonzero(pass)
    //
    // preserve original anchor order
    //
    // overflow = number_of_candidates > MAX_CAND
    //
    // cand = cand[:MAX_CAND]
    // ============================================================

    candidate_collector #(
        .COORD_BITS(
            COORD_BITS
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
        )
    ) u_candidate_collector (
        .clk(clk),
        .rst_n(rst_n),

        .frame_start(
            frame_start
        ),

        .frame_end(
            frame_end
        ),

        .in_valid(
            in_valid
        ),

        .x1(in_x1),
        .y1(in_y1),
        .x2(in_x2),
        .y2(in_y2),

        .cls(in_cls),

        .score(
            in_score
        ),

        .pass(
            in_pass
        ),

        .anchor_idx(
            in_anchor_idx
        ),

        .cand_count(
            cand_count
        ),

        .overflow(
            overflow
        ),

        .collect_done(
            collect_done
        ),

        .rd_addr(
            cand_rd_addr
        ),

        .rd_x1(
            cand_x1
        ),

        .rd_y1(
            cand_y1
        ),

        .rd_x2(
            cand_x2
        ),

        .rd_y2(
            cand_y2
        ),

        .rd_cls(
            cand_cls
        ),

        .rd_score(
            cand_score
        ),

        .rd_anchor(
            cand_anchor
        )
    );


    // ============================================================
    // NMS start generation
    //
    // candidate_collector asserts collect_done after collection.
    //
    // Delay by one clock before starting NMS so that:
    //
    // cand_count
    // overflow
    // Candidate RAM
    //
    // are all stable.
    // ============================================================

    logic nms_start;


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            nms_start <=
                1'b0;

        end
        else begin

            nms_start <=
                collect_done;

        end

    end


    // ============================================================
    // NMS Engine
    //
    // Golden ordering:
    //
    // class ascending
    //
    // then:
    //
    // score descending
    //
    // then:
    //
    // anchor index ascending on score tie
    //
    // Followed by class-aware greedy IoU suppression.
    // ============================================================

    nms_engine #(
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
    ) u_nms_engine (
        .clk(clk),
        .rst_n(rst_n),

        .start(
            nms_start
        ),

        .cand_count(
            cand_count
        ),

        .cand_rd_addr(
            cand_rd_addr
        ),

        .cand_x1(
            cand_x1
        ),

        .cand_y1(
            cand_y1
        ),

        .cand_x2(
            cand_x2
        ),

        .cand_y2(
            cand_y2
        ),

        .cand_cls(
            cand_cls
        ),

        .cand_score(
            cand_score
        ),

        .cand_anchor(
            cand_anchor
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

        .det_anchor(
            det_anchor_idx
        ),

        .n_det(
            n_det
        ),

        .done(
            done
        )
    );


    // ============================================================
    // Simulation-only configuration checks
    // ============================================================

`ifndef SYNTHESIS

    initial begin

        if (
            NUM_CLASSES < 1
        ) begin

            $error(
                "yolov8_nms_top: NUM_CLASSES must be >= 1"
            );

        end


        if (
            (1 << CLASS_BITS) <
            NUM_CLASSES
        ) begin

            $error(
                "yolov8_nms_top: CLASS_BITS too small for NUM_CLASSES"
            );

        end


        if (
            MAX_CAND < 1
        ) begin

            $error(
                "yolov8_nms_top: MAX_CAND must be >= 1"
            );

        end


        if (
            MAX_DET < 1
        ) begin

            $error(
                "yolov8_nms_top: MAX_DET must be >= 1"
            );

        end


        if (
            IOU_DEN <= 0
        ) begin

            $error(
                "yolov8_nms_top: IOU_DEN must be > 0"
            );

        end


        if (
            IOU_NUM < 0
        ) begin

            $error(
                "yolov8_nms_top: IOU_NUM must be >= 0"
            );

        end


        if (
            IOU_NUM >
            IOU_DEN
        ) begin

            $error(
                "yolov8_nms_top: IoU threshold must be <= 1"
            );

        end

    end


    always_ff @(posedge clk) begin

        if (rst_n) begin

            if (
                cand_count >
                MAX_CAND
            ) begin

                $error(
                    "yolov8_nms_top: cand_count exceeds MAX_CAND"
                );

            end


            if (
                n_det >
                MAX_DET
            ) begin

                $error(
                    "yolov8_nms_top: n_det exceeds MAX_DET"
                );

            end

        end

    end

`endif

endmodule