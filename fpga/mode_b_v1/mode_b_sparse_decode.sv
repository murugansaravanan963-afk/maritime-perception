module mode_b_sparse_decode #(
    parameter integer COORD_BITS  = 14,
    parameter integer OFF_BITS    = 12,
    parameter integer ANCHOR_BITS = 15,
    parameter integer CLASS_BITS  = 3,
    parameter integer SCORE_BITS  = 8,
    parameter integer IMG_W       = 960,
    parameter integer IMG_H       = 960
)(
    input  logic clk,
    input  logic rst_n,

    input  logic in_valid,

    input  logic [ANCHOR_BITS-1:0] anchor_idx,
    input  logic [CLASS_BITS-1:0]  in_cls,
    input  logic [SCORE_BITS-1:0]  in_score,

    input  logic [OFF_BITS-1:0] off_l,
    input  logic [OFF_BITS-1:0] off_t,
    input  logic [OFF_BITS-1:0] off_r,
    input  logic [OFF_BITS-1:0] off_b,

    output logic out_valid,

    output logic [COORD_BITS-1:0] x1,
    output logic [COORD_BITS-1:0] y1,
    output logic [COORD_BITS-1:0] x2,
    output logic [COORD_BITS-1:0] y2,

    output logic [ANCHOR_BITS-1:0] out_anchor_idx,
    output logic [CLASS_BITS-1:0]  out_cls,
    output logic [SCORE_BITS-1:0]  out_score
);

    localparam integer G8W     = IMG_W / 8;   // 120
    localparam integer G8H     = IMG_H / 8;   // 120
    localparam integer G16W    = IMG_W / 16;  // 60
    localparam integer G16H    = IMG_H / 16;  // 60
    localparam integer G32W    = IMG_W / 32;  // 30

    localparam integer START16 = G8W * G8H;          // 14400
    localparam integer START32 = START16 +
                                 G16W * G16H;        // 18000

    localparam integer MAX_X_Q4 = IMG_W << 4;
    localparam integer MAX_Y_Q4 = IMG_H << 4;


    // ============================================================
    // Stage 1A
    //
    // anchor_idx
    //   -> scale selection
    //   -> local index
    //   -> exact constant division for gy
    //   -> register
    //
    // gx is intentionally NOT calculated here.
    // This breaks the previous critical path:
    //
    // anchor -> gy -> gx -> register
    // ============================================================

    logic s1a_valid;

    logic [ANCHOR_BITS-1:0] s1a_anchor;
    logic [CLASS_BITS-1:0]  s1a_cls;
    logic [SCORE_BITS-1:0]  s1a_score;

    logic [OFF_BITS-1:0] s1a_off_l;
    logic [OFF_BITS-1:0] s1a_off_t;
    logic [OFF_BITS-1:0] s1a_off_r;
    logic [OFF_BITS-1:0] s1a_off_b;

    logic [14:0] s1a_local_idx;
    logic [7:0]  s1a_gy;
    logic [2:0]  s1a_lg;

    integer local_idx_c;
    integer gy_c;
    integer lg_c;

    always_comb begin

        local_idx_c = 0;
        gy_c        = 0;
        lg_c        = 3;

        if (anchor_idx < START16) begin

            local_idx_c = anchor_idx;
            lg_c = 3;

            // Exact over the legal Maritime960 range:
            //
            // floor(n / 120)
            //
            // 17477 / 2^21
            gy_c = (local_idx_c * 17477) >> 21;

        end
        else if (anchor_idx < START32) begin

            local_idx_c = anchor_idx - START16;
            lg_c = 4;

            // Exact over the legal Maritime960 range:
            //
            // floor(n / 60)
            gy_c = (local_idx_c * 2185) >> 17;

        end
        else begin

            local_idx_c = anchor_idx - START32;
            lg_c = 5;

            // Exact over the legal Maritime960 range:
            //
            // floor(n / 30)
            gy_c = (local_idx_c * 1093) >> 15;

        end

    end


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            s1a_valid     <= 1'b0;

            s1a_anchor    <= '0;
            s1a_cls       <= '0;
            s1a_score     <= '0;

            s1a_off_l     <= '0;
            s1a_off_t     <= '0;
            s1a_off_r     <= '0;
            s1a_off_b     <= '0;

            s1a_local_idx <= '0;
            s1a_gy        <= '0;
            s1a_lg        <= '0;

        end
        else begin

            s1a_valid <= in_valid;

            if (in_valid) begin

                s1a_anchor <= anchor_idx;
                s1a_cls    <= in_cls;
                s1a_score  <= in_score;

                s1a_off_l <= off_l;
                s1a_off_t <= off_t;
                s1a_off_r <= off_r;
                s1a_off_b <= off_b;

                s1a_local_idx <= local_idx_c[14:0];
                s1a_gy        <= gy_c[7:0];
                s1a_lg        <= lg_c[2:0];

            end

        end

    end


    // ============================================================
    // Stage 1B
    //
    // registered local_idx + registered gy
    //              ->
    //             gx
    //              ->
    //           register
    //
    // This is the stage specifically added for MAX 10 timing.
    // ============================================================

    logic s1b_valid;

    logic [ANCHOR_BITS-1:0] s1b_anchor;
    logic [CLASS_BITS-1:0]  s1b_cls;
    logic [SCORE_BITS-1:0]  s1b_score;

    logic [OFF_BITS-1:0] s1b_off_l;
    logic [OFF_BITS-1:0] s1b_off_t;
    logic [OFF_BITS-1:0] s1b_off_r;
    logic [OFF_BITS-1:0] s1b_off_b;

    logic [7:0] s1b_gx;
    logic [7:0] s1b_gy;
    logic [2:0] s1b_lg;

    integer gx_c;

    always_comb begin

        gx_c = 0;

        case (s1a_lg)

            3:
                gx_c =
                    s1a_local_idx -
                    s1a_gy * G8W;

            4:
                gx_c =
                    s1a_local_idx -
                    s1a_gy * G16W;

            default:
                gx_c =
                    s1a_local_idx -
                    s1a_gy * G32W;

        endcase

    end


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            s1b_valid  <= 1'b0;

            s1b_anchor <= '0;
            s1b_cls    <= '0;
            s1b_score  <= '0;

            s1b_off_l  <= '0;
            s1b_off_t  <= '0;
            s1b_off_r  <= '0;
            s1b_off_b  <= '0;

            s1b_gx     <= '0;
            s1b_gy     <= '0;
            s1b_lg     <= '0;

        end
        else begin

            s1b_valid <= s1a_valid;

            if (s1a_valid) begin

                s1b_anchor <= s1a_anchor;
                s1b_cls    <= s1a_cls;
                s1b_score  <= s1a_score;

                s1b_off_l <= s1a_off_l;
                s1b_off_t <= s1a_off_t;
                s1b_off_r <= s1a_off_r;
                s1b_off_b <= s1a_off_b;

                s1b_gx <= gx_c[7:0];
                s1b_gy <= s1a_gy;
                s1b_lg <= s1a_lg;

            end

        end

    end


    // ============================================================
    // Stage 2
    //
    // gx / gy
    //   -> anchor center
    //   -> DFL offset * stride
    //   -> raw bbox
    //   -> register
    // ============================================================

    logic s2_valid;

    logic [ANCHOR_BITS-1:0] s2_anchor;
    logic [CLASS_BITS-1:0]  s2_cls;
    logic [SCORE_BITS-1:0]  s2_score;

    logic signed [COORD_BITS+3:0] s2_tx1;
    logic signed [COORD_BITS+3:0] s2_ty1;
    logic signed [COORD_BITS+3:0] s2_tx2;
    logic signed [COORD_BITS+3:0] s2_ty2;

    integer cx_c;
    integer cy_c;

    integer opl_c;
    integer opt_c;
    integer opr_c;
    integer opb_c;

    integer tx1_c;
    integer ty1_c;
    integer tx2_c;
    integer ty2_c;

    always_comb begin

        // Q.4 anchor center
        cx_c =
            (2 * s1b_gx + 1) *
            (1 << s1b_lg) * 8;

        cy_c =
            (2 * s1b_gy + 1) *
            (1 << s1b_lg) * 8;


        // Q4.8 DFL offset
        // -> Q.4 coordinate offset

        opl_c = (s1b_off_l << s1b_lg) >> 4;
        opt_c = (s1b_off_t << s1b_lg) >> 4;
        opr_c = (s1b_off_r << s1b_lg) >> 4;
        opb_c = (s1b_off_b << s1b_lg) >> 4;


        tx1_c = cx_c - opl_c;
        ty1_c = cy_c - opt_c;

        tx2_c = cx_c + opr_c;
        ty2_c = cy_c + opb_c;

    end


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            s2_valid <= 1'b0;

            s2_anchor <= '0;
            s2_cls    <= '0;
            s2_score  <= '0;

            s2_tx1 <= '0;
            s2_ty1 <= '0;
            s2_tx2 <= '0;
            s2_ty2 <= '0;

        end
        else begin

            s2_valid <= s1b_valid;

            if (s1b_valid) begin

                s2_anchor <= s1b_anchor;
                s2_cls    <= s1b_cls;
                s2_score  <= s1b_score;

                s2_tx1 <= tx1_c;
                s2_ty1 <= ty1_c;
                s2_tx2 <= tx2_c;
                s2_ty2 <= ty2_c;

            end

        end

    end


    // ============================================================
    // Stage 3
    //
    // Clamp
    //   -> registered output
    // ============================================================

    integer clamp_x1_c;
    integer clamp_y1_c;
    integer clamp_x2_c;
    integer clamp_y2_c;

    always_comb begin

        if (s2_tx1 < 0)
            clamp_x1_c = 0;
        else if (s2_tx1 > MAX_X_Q4)
            clamp_x1_c = MAX_X_Q4;
        else
            clamp_x1_c = s2_tx1;


        if (s2_ty1 < 0)
            clamp_y1_c = 0;
        else if (s2_ty1 > MAX_Y_Q4)
            clamp_y1_c = MAX_Y_Q4;
        else
            clamp_y1_c = s2_ty1;


        if (s2_tx2 < 0)
            clamp_x2_c = 0;
        else if (s2_tx2 > MAX_X_Q4)
            clamp_x2_c = MAX_X_Q4;
        else
            clamp_x2_c = s2_tx2;


        if (s2_ty2 < 0)
            clamp_y2_c = 0;
        else if (s2_ty2 > MAX_Y_Q4)
            clamp_y2_c = MAX_Y_Q4;
        else
            clamp_y2_c = s2_ty2;

    end


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            out_valid <= 1'b0;

            x1 <= '0;
            y1 <= '0;
            x2 <= '0;
            y2 <= '0;

            out_anchor_idx <= '0;
            out_cls        <= '0;
            out_score      <= '0;

        end
        else begin

            out_valid <= s2_valid;

            if (s2_valid) begin

                x1 <= clamp_x1_c[COORD_BITS-1:0];
                y1 <= clamp_y1_c[COORD_BITS-1:0];
                x2 <= clamp_x2_c[COORD_BITS-1:0];
                y2 <= clamp_y2_c[COORD_BITS-1:0];

                out_anchor_idx <= s2_anchor;
                out_cls        <= s2_cls;
                out_score      <= s2_score;

            end

        end

    end

endmodule