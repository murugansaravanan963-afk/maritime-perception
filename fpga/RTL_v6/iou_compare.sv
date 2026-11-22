`timescale 1ns / 1ps

module iou_compare #(
    // ============================================================
    // Coordinate format
    // ============================================================

    parameter integer COORD_BITS = 14,

    // ============================================================
    // IoU threshold
    //
    // Suppress when:
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
    // This is MODEL-DEPENDENT and should ultimately come from
    // rtl_params.vh.
    //
    // Current generated configuration:
    //
    // AREA_BITS = 28
    // ============================================================

    parameter integer AREA_BITS = 28
)(
    input logic clk,
    input logic rst_n,

    input logic in_valid,

    // ============================================================
    // Box A
    // ============================================================

    input logic [COORD_BITS-1:0] ax1,
    input logic [COORD_BITS-1:0] ay1,
    input logic [COORD_BITS-1:0] ax2,
    input logic [COORD_BITS-1:0] ay2,

    // ============================================================
    // Box B
    // ============================================================

    input logic [COORD_BITS-1:0] bx1,
    input logic [COORD_BITS-1:0] by1,
    input logic [COORD_BITS-1:0] bx2,
    input logic [COORD_BITS-1:0] by2,

    // ============================================================
    // Result
    // ============================================================

    output logic out_valid,

    output logic suppress
);

    // ============================================================
    // Derived widths
    // ============================================================

    // Coordinate difference has the same unsigned width as the
    // clipped coordinate.

    localparam integer DIFF_BITS =
        COORD_BITS;


    // Union can be:
    //
    // areaA + areaB - intersection
    //
    // so it needs one extra bit beyond AREA_BITS.

    localparam integer UNION_BITS =
        AREA_BITS + 1;


    // Determine how many bits are required to represent the
    // larger IoU constant.

    localparam integer MAX_IOU_CONST =
        (IOU_NUM > IOU_DEN) ?
            IOU_NUM :
            IOU_DEN;


    localparam integer IOU_CONST_BITS =
        (MAX_IOU_CONST <= 1) ?
            1 :
            $clog2(MAX_IOU_CONST + 1);


    // Threshold comparison operands:
    //
    // lhs = IOU_DEN * inter
    // rhs = IOU_NUM * union
    //
    // rhs uses UNION_BITS, so size both sides from that worst case.

    localparam integer COMP_BITS =
        UNION_BITS +
        IOU_CONST_BITS;


    // ============================================================
    // Stage 1
    //
    // Intersection boundaries:
    //
    // ix1 = max(ax1, bx1)
    // iy1 = max(ay1, by1)
    // ix2 = min(ax2, bx2)
    // iy2 = min(ay2, by2)
    //
    // Also calculate A/B dimensions.
    // ============================================================

    logic [COORD_BITS-1:0] ix1_s1;
    logic [COORD_BITS-1:0] iy1_s1;
    logic [COORD_BITS-1:0] ix2_s1;
    logic [COORD_BITS-1:0] iy2_s1;

    logic [DIFF_BITS-1:0] aw_s1;
    logic [DIFF_BITS-1:0] ah_s1;

    logic [DIFF_BITS-1:0] bw_s1;
    logic [DIFF_BITS-1:0] bh_s1;

    logic valid_s1;


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            ix1_s1 <= '0;
            iy1_s1 <= '0;
            ix2_s1 <= '0;
            iy2_s1 <= '0;

            aw_s1 <= '0;
            ah_s1 <= '0;

            bw_s1 <= '0;
            bh_s1 <= '0;

            valid_s1 <= 1'b0;

        end
        else begin

            valid_s1 <= in_valid;


            if (in_valid) begin

                // max(x1)

                if (ax1 >= bx1)
                    ix1_s1 <= ax1;
                else
                    ix1_s1 <= bx1;


                // max(y1)

                if (ay1 >= by1)
                    iy1_s1 <= ay1;
                else
                    iy1_s1 <= by1;


                // min(x2)

                if (ax2 <= bx2)
                    ix2_s1 <= ax2;
                else
                    ix2_s1 <= bx2;


                // min(y2)

                if (ay2 <= by2)
                    iy2_s1 <= ay2;
                else
                    iy2_s1 <= by2;


                // ------------------------------------------------
                // Box A dimensions
                // ------------------------------------------------

                if (ax2 > ax1)
                    aw_s1 <= ax2 - ax1;
                else
                    aw_s1 <= '0;


                if (ay2 > ay1)
                    ah_s1 <= ay2 - ay1;
                else
                    ah_s1 <= '0;


                // ------------------------------------------------
                // Box B dimensions
                // ------------------------------------------------

                if (bx2 > bx1)
                    bw_s1 <= bx2 - bx1;
                else
                    bw_s1 <= '0;


                if (by2 > by1)
                    bh_s1 <= by2 - by1;
                else
                    bh_s1 <= '0;

            end

        end

    end


    // ============================================================
    // Stage 2
    //
    // Intersection width / height
    // ============================================================

    logic [DIFF_BITS-1:0] iw_s2;
    logic [DIFF_BITS-1:0] ih_s2;

    logic [DIFF_BITS-1:0] aw_s2;
    logic [DIFF_BITS-1:0] ah_s2;

    logic [DIFF_BITS-1:0] bw_s2;
    logic [DIFF_BITS-1:0] bh_s2;

    logic valid_s2;


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            iw_s2 <= '0;
            ih_s2 <= '0;

            aw_s2 <= '0;
            ah_s2 <= '0;

            bw_s2 <= '0;
            bh_s2 <= '0;

            valid_s2 <= 1'b0;

        end
        else begin

            valid_s2 <= valid_s1;


            if (valid_s1) begin

                if (ix2_s1 > ix1_s1)
                    iw_s2 <= ix2_s1 - ix1_s1;
                else
                    iw_s2 <= '0;


                if (iy2_s1 > iy1_s1)
                    ih_s2 <= iy2_s1 - iy1_s1;
                else
                    ih_s2 <= '0;


                aw_s2 <= aw_s1;
                ah_s2 <= ah_s1;

                bw_s2 <= bw_s1;
                bh_s2 <= bh_s1;

            end

        end

    end


    // ============================================================
    // Stage 3
    //
    // Areas
    //
    // Because all coordinates have the same Q.COORD_FRAC scale,
    // the common squared fixed-point scale cancels in the IoU
    // comparison.
    // ============================================================

    logic [AREA_BITS-1:0] inter_s3;

    logic [AREA_BITS-1:0] area_a_s3;
    logic [AREA_BITS-1:0] area_b_s3;

    logic valid_s3;


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            inter_s3 <= '0;

            area_a_s3 <= '0;
            area_b_s3 <= '0;

            valid_s3 <= 1'b0;

        end
        else begin

            valid_s3 <= valid_s2;


            if (valid_s2) begin

                inter_s3 <=
                    iw_s2 * ih_s2;

                area_a_s3 <=
                    aw_s2 * ah_s2;

                area_b_s3 <=
                    bw_s2 * bh_s2;

            end

        end

    end


    // ============================================================
    // Stage 4
    //
    // union = areaA + areaB - intersection
    // ============================================================

    logic [UNION_BITS-1:0]
        union_s4;

    logic [AREA_BITS-1:0]
        inter_s4;

    logic valid_s4;


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            union_s4 <= '0;
            inter_s4 <= '0;

            valid_s4 <= 1'b0;

        end
        else begin

            valid_s4 <= valid_s3;


            if (valid_s3) begin

                union_s4 <=
                    {
                        1'b0,
                        area_a_s3
                    }
                    +
                    {
                        1'b0,
                        area_b_s3
                    }
                    -
                    {
                        1'b0,
                        inter_s3
                    };


                inter_s4 <=
                    inter_s3;

            end

        end

    end


    // ============================================================
    // Stage 5 preparation
    //
    // Explicitly extend operands before multiplying.
    // ============================================================

    logic [COMP_BITS-1:0]
        inter_ext;

    logic [COMP_BITS-1:0]
        union_ext;

    logic [COMP_BITS-1:0]
        lhs;

    logic [COMP_BITS-1:0]
        rhs;


    always_comb begin

        inter_ext = '0;
        union_ext = '0;


        inter_ext[
            AREA_BITS-1:0
        ] = inter_s4;


        union_ext[
            UNION_BITS-1:0
        ] = union_s4;


        lhs =
            inter_ext *
            IOU_DEN;


        rhs =
            union_ext *
            IOU_NUM;

    end


    // ============================================================
    // Stage 5
    //
    // Golden rule:
    //
    // suppress =
    //
    // inter > 0
    //
    // AND
    //
    // IOU_DEN * inter >= IOU_NUM * union
    // ============================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            suppress <= 1'b0;

            out_valid <= 1'b0;

        end
        else begin

            out_valid <= valid_s4;


            if (valid_s4) begin

                if (
                    (inter_s4 != 0)
                    &&
                    (lhs >= rhs)
                )
                    suppress <= 1'b1;
                else
                    suppress <= 1'b0;

            end

        end

    end


    // ============================================================
    // Simulation-only configuration checks
    // ============================================================

`ifndef SYNTHESIS

    initial begin

        if (COORD_BITS < 1) begin

            $error(
                "iou_compare: COORD_BITS must be >= 1"
            );

        end

        if (IOU_NUM < 0) begin

            $error(
                "iou_compare: IOU_NUM must be >= 0"
            );

        end


        if (IOU_DEN <= 0) begin

            $error(
                "iou_compare: IOU_DEN must be > 0"
            );

        end


        if (
            IOU_NUM >
            IOU_DEN
        ) begin

            $error(
                "iou_compare: expected IoU threshold must be <= 1"
            );

        end

    end

`endif

endmodule