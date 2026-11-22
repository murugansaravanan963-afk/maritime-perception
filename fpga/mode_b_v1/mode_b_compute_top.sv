module mode_b_compute_top #(
    parameter integer Q_BITS=8,
    parameter integer DFL_BINS=16,
    parameter integer DFL_SIDES=4,
    parameter integer LUT_WIDTH=16,
    parameter integer LUT_DEPTH=65,

    parameter LUT_FILE="dfl_exp_lut.mem",

    parameter integer OFF_BITS=12,

    parameter integer NUM_CLASSES=6,
    parameter integer SCORE_BITS=8,
    parameter integer CLASS_BITS=3,

    parameter integer COORD_BITS=14,
    parameter integer ANCHOR_BITS=15,

    parameter integer IMG_W=960,
    parameter integer IMG_H=960,

    parameter integer MAX_CAND=4096,
    parameter integer MAX_DET=100,

    parameter integer IOU_NUM=9,
    parameter integer IOU_DEN=20,

    parameter integer AREA_BITS=28
)(
    input logic clk,
    input logic rst_n,

    input logic frame_start,

    input logic [15:0] n_candidates,

    input logic in_valid,
    output logic in_ready,

    input logic [ANCHOR_BITS-1:0] in_anchor_idx,
    input logic [CLASS_BITS-1:0]  in_cls,
    input logic [SCORE_BITS-1:0]  in_score,

    input logic [Q_BITS-1:0]
        in_bbox [0:DFL_SIDES*DFL_BINS-1],

    output logic det_valid,

    output logic [COORD_BITS-1:0] det_x1,
    output logic [COORD_BITS-1:0] det_y1,
    output logic [COORD_BITS-1:0] det_x2,
    output logic [COORD_BITS-1:0] det_y2,

    output logic [CLASS_BITS-1:0] det_cls,
    output logic [SCORE_BITS-1:0] det_score,

    output logic [ANCHOR_BITS-1:0] det_anchor_idx,

    output logic [$clog2(MAX_DET+1)-1:0] n_det,

    output logic overflow,
    output logic done
);

    // ============================================================
    // DFL interface
    // ============================================================

    logic dfl_valid;
    logic dfl_ready;

    logic [OFF_BITS-1:0] off_l;
    logic [OFF_BITS-1:0] off_t;
    logic [OFF_BITS-1:0] off_r;
    logic [OFF_BITS-1:0] off_b;


    // ============================================================
    // Metadata FIFO
    //
    // Metadata remains aligned with each DFL transaction.
    // ============================================================

    localparam integer META_DEPTH = 4;

    logic [ANCHOR_BITS-1:0]
        meta_anchor [0:META_DEPTH-1];

    logic [CLASS_BITS-1:0]
        meta_cls [0:META_DEPTH-1];

    logic [SCORE_BITS-1:0]
        meta_score [0:META_DEPTH-1];

    logic [2:0] wrp;
    logic [2:0] rdp;

    logic [15:0] accepted;
    logic [15:0] retired;


    wire meta_full =
        (wrp[1:0] == rdp[1:0]) &&
        (wrp[2]   != rdp[2]);


    assign in_ready =
        dfl_ready &&
        !meta_full &&
        (accepted < n_candidates);


    wire accept =
        in_valid &&
        in_ready;


    // ============================================================
    // Dual-bank / 4-side parallel DFL
    // ============================================================

    yolov8_dfl_top #(
        .Q_BITS(Q_BITS),
        .DFL_BINS(DFL_BINS),
        .DFL_SIDES(DFL_SIDES),

        .LUT_WIDTH(LUT_WIDTH),
        .LUT_DEPTH(LUT_DEPTH),
        .LUT_FILE(LUT_FILE),

        .OFF_BITS(OFF_BITS)
    ) u_dfl (
        .clk(clk),
        .rst_n(rst_n),

        .in_valid(accept),
        .in_ready(dfl_ready),

        .bbox_q(in_bbox),

        .out_valid(dfl_valid),

        .off_l(off_l),
        .off_t(off_t),
        .off_r(off_r),
        .off_b(off_b)
    );


    // ============================================================
    // Metadata / DFL-result register stage
    //
    // Previous critical path:
    //
    // rdp
    //  -> meta_anchor[rdp]
    //  -> sparse decode scale selection
    //  -> local_idx
    //  -> constant arithmetic
    //  -> s1a_gy
    //
    // This stage breaks that path at the metadata FIFO output.
    //
    // IMPORTANT:
    // DFL offsets are registered together with metadata so that
    // anchor/class/score and DFL results remain bit-exact aligned.
    // ============================================================

    logic meta_retire_valid;

    logic [ANCHOR_BITS-1:0] meta_retire_anchor;
    logic [CLASS_BITS-1:0]  meta_retire_cls;
    logic [SCORE_BITS-1:0]  meta_retire_score;

    logic [OFF_BITS-1:0] meta_retire_off_l;
    logic [OFF_BITS-1:0] meta_retire_off_t;
    logic [OFF_BITS-1:0] meta_retire_off_r;
    logic [OFF_BITS-1:0] meta_retire_off_b;


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            meta_retire_valid  <= 1'b0;

            meta_retire_anchor <= '0;
            meta_retire_cls    <= '0;
            meta_retire_score  <= '0;

            meta_retire_off_l  <= '0;
            meta_retire_off_t  <= '0;
            meta_retire_off_r  <= '0;
            meta_retire_off_b  <= '0;

        end
        else begin

            meta_retire_valid <= dfl_valid;

            if (dfl_valid) begin

                // Capture metadata selected by the current
                // retirement pointer.
                meta_retire_anchor <= meta_anchor[rdp[1:0]];
                meta_retire_cls    <= meta_cls[rdp[1:0]];
                meta_retire_score  <= meta_score[rdp[1:0]];

                // Capture the corresponding DFL result in the
                // same clock so metadata and offsets remain aligned.
                meta_retire_off_l <= off_l;
                meta_retire_off_t <= off_t;
                meta_retire_off_r <= off_r;
                meta_retire_off_b <= off_b;

            end

        end

    end


    // ============================================================
    // MAX 10 optimized sparse decode
    //
    // Input is now REGISTERED metadata + REGISTERED DFL offsets.
    //
    // Sparse decode internal pipeline:
    //
    // Stage 1A:
    //   scale / local_idx / gy
    //
    // Stage 1B:
    //   gx
    //
    // Stage 2:
    //   center + DFL offset + raw bbox
    //
    // Stage 3:
    //   clamp + output
    // ============================================================

    logic dec_valid;

    logic [COORD_BITS-1:0] dec_x1;
    logic [COORD_BITS-1:0] dec_y1;
    logic [COORD_BITS-1:0] dec_x2;
    logic [COORD_BITS-1:0] dec_y2;

    logic [ANCHOR_BITS-1:0] dec_anchor;
    logic [CLASS_BITS-1:0]  dec_cls;
    logic [SCORE_BITS-1:0]  dec_score;


    mode_b_sparse_decode #(
        .COORD_BITS(COORD_BITS),
        .OFF_BITS(OFF_BITS),

        .ANCHOR_BITS(ANCHOR_BITS),

        .CLASS_BITS(CLASS_BITS),
        .SCORE_BITS(SCORE_BITS),

        .IMG_W(IMG_W),
        .IMG_H(IMG_H)
    ) u_dec (
        .clk(clk),
        .rst_n(rst_n),

        .in_valid(meta_retire_valid),

        .anchor_idx(meta_retire_anchor),

        .in_cls(meta_retire_cls),
        .in_score(meta_retire_score),

        .off_l(meta_retire_off_l),
        .off_t(meta_retire_off_t),
        .off_r(meta_retire_off_r),
        .off_b(meta_retire_off_b),

        .out_valid(dec_valid),

        .x1(dec_x1),
        .y1(dec_y1),
        .x2(dec_x2),
        .y2(dec_y2),

        .out_anchor_idx(dec_anchor),
        .out_cls(dec_cls),
        .out_score(dec_score)
    );


    // ============================================================
    // Last-candidate alignment
    //
    // Pipeline after dfl_valid is now:
    //
    //   Metadata register : 1 clock
    //   Sparse Stage 1A   : 1 clock
    //   Sparse Stage 1B   : 1 clock
    //   Sparse Stage 2    : 1 clock
    //   Sparse Stage 3    : 1 clock
    //
    // Total = 5 clocks.
    //
    // Therefore dfl_last must be delayed by five clocks before
    // frame_end is presented to NMS.
    // ============================================================

    logic last_s1;
    logic last_s2;
    logic last_s3;
    logic last_s4;
    logic last_s5;


    wire dfl_last =
        dfl_valid &&
        (retired + 1'b1 == n_candidates);


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            last_s1 <= 1'b0;
            last_s2 <= 1'b0;
            last_s3 <= 1'b0;
            last_s4 <= 1'b0;
            last_s5 <= 1'b0;

        end
        else begin

            last_s1 <= dfl_last;
            last_s2 <= last_s1;
            last_s3 <= last_s2;
            last_s4 <= last_s3;
            last_s5 <= last_s4;

        end

    end


    // ============================================================
    // Empty Mode-B frame
    // ============================================================

    logic empty_frame_end;


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            empty_frame_end <= 1'b0;

        end
        else begin

            empty_frame_end <= 1'b0;

            if (
                frame_start &&
                (n_candidates == 16'd0)
            ) begin

                empty_frame_end <= 1'b1;

            end

        end

    end


    wire nms_frame_end =
        last_s5 |
        empty_frame_end;


    // ============================================================
    // Class-aware integer NMS
    // ============================================================

    yolov8_nms_top #(
        .COORD_BITS(COORD_BITS),

        .NUM_CLASSES(NUM_CLASSES),
        .CLASS_BITS(CLASS_BITS),

        .SCORE_BITS(SCORE_BITS),

        .ANCHOR_BITS(ANCHOR_BITS),

        .MAX_CAND(MAX_CAND),
        .MAX_DET(MAX_DET),

        .IOU_NUM(IOU_NUM),
        .IOU_DEN(IOU_DEN),

        .AREA_BITS(AREA_BITS)
    ) u_nms (
        .clk(clk),
        .rst_n(rst_n),

        .frame_start(frame_start),
        .frame_end(nms_frame_end),

        .in_valid(dec_valid),

        .in_x1(dec_x1),
        .in_y1(dec_y1),
        .in_x2(dec_x2),
        .in_y2(dec_y2),

        .in_cls(dec_cls),
        .in_score(dec_score),

        // Mode-B candidates have already passed host-side
        // class/score prefiltering.
        .in_pass(1'b1),

        .in_anchor_idx(dec_anchor),

        .det_valid(det_valid),

        .det_x1(det_x1),
        .det_y1(det_y1),
        .det_x2(det_x2),
        .det_y2(det_y2),

        .det_cls(det_cls),
        .det_score(det_score),

        .det_anchor_idx(det_anchor_idx),

        .n_det(n_det),

        .overflow(overflow),
        .done(done)
    );


    // ============================================================
    // Metadata FIFO / transaction accounting
    // ============================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            wrp <= '0;
            rdp <= '0;

            accepted <= 16'd0;
            retired  <= 16'd0;

        end
        else begin

            if (frame_start) begin

                wrp <= '0;
                rdp <= '0;

                accepted <= 16'd0;
                retired  <= 16'd0;

            end
            else begin

                // ------------------------------------------------
                // New candidate accepted by DFL
                // ------------------------------------------------

                if (accept) begin

                    meta_anchor[wrp[1:0]] <= in_anchor_idx;
                    meta_cls[wrp[1:0]]    <= in_cls;
                    meta_score[wrp[1:0]]  <= in_score;

                    wrp <= wrp + 1'b1;

                    accepted <= accepted + 1'b1;

                end


                // ------------------------------------------------
                // Ordered DFL retirement
                // ------------------------------------------------

                if (dfl_valid) begin

                    rdp <= rdp + 1'b1;

                    retired <= retired + 1'b1;

                end

            end

        end

    end

endmodule