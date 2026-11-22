`timescale 1ns / 1ps

module yolov8_dfl_top #(
    parameter integer Q_BITS = 8,

    parameter integer DFL_BINS  = 16,
    parameter integer DFL_SIDES = 4,

    parameter integer LUT_WIDTH = 16,
    parameter integer LUT_DEPTH = 65,

    parameter LUT_FILE =
        "dfl_exp_lut.mem",

    parameter integer OFF_BITS = 12,
    parameter integer OFF_FRAC = 8
)(
    input logic clk,
    input logic rst_n,

    // ============================================================
    // Input handshake
    // ============================================================

    input logic in_valid,

    output logic in_ready,

    input logic [Q_BITS-1:0]
        bbox_q [0:DFL_SIDES*DFL_BINS-1],

    // ============================================================
    // Ordered output
    // ============================================================

    output logic out_valid,

    output logic [OFF_BITS-1:0] off_l,
    output logic [OFF_BITS-1:0] off_t,
    output logic [OFF_BITS-1:0] off_r,
    output logic [OFF_BITS-1:0] off_b
);


    // ============================================================
    // V5 architecture
    //
    //               Anchor Stream
    //                    |
    //             +------+------+
    //             |             |
    //           Bank A        Bank B
    //          L T R B        L T R B
    //             |             |
    //             +------+------+
    //                    |
    //             Ordered retire
    //                    |
    //                 Output
    //
    // Each bank contains four parallel DFL cores.
    //
    // Two anchors may therefore be in flight simultaneously.
    //
    // A two-entry bank-ID FIFO preserves original input order.
    // ============================================================


    localparam integer TOTAL_BINS =
        DFL_SIDES * DFL_BINS;


    // ============================================================
    // BANK A state
    // ============================================================

    logic bank_a_busy;
    logic bank_a_start_pending;
    logic bank_a_result_ready;

    logic [Q_BITS-1:0]
        bank_a_bbox [0:TOTAL_BINS-1];

    logic [OFF_BITS-1:0]
        bank_a_off_l;

    logic [OFF_BITS-1:0]
        bank_a_off_t;

    logic [OFF_BITS-1:0]
        bank_a_off_r;

    logic [OFF_BITS-1:0]
        bank_a_off_b;


    // ============================================================
    // BANK B state
    // ============================================================

    logic bank_b_busy;
    logic bank_b_start_pending;
    logic bank_b_result_ready;

    logic [Q_BITS-1:0]
        bank_b_bbox [0:TOTAL_BINS-1];

    logic [OFF_BITS-1:0]
        bank_b_off_l;

    logic [OFF_BITS-1:0]
        bank_b_off_t;

    logic [OFF_BITS-1:0]
        bank_b_off_r;

    logic [OFF_BITS-1:0]
        bank_b_off_b;


    // ============================================================
    // Input arbitration
    //
    // Prefer A when both banks are free.
    // ============================================================

    logic select_bank_a;
    logic select_bank_b;

    always_comb begin

        select_bank_a =
            1'b0;

        select_bank_b =
            1'b0;


        if (!bank_a_busy) begin

            select_bank_a =
                1'b1;

        end

        else if (!bank_b_busy) begin

            select_bank_b =
                1'b1;

        end

    end


    always_comb begin

        in_ready =
            !bank_a_busy ||
            !bank_b_busy;

    end


    wire input_fire =
        in_valid &&
        in_ready;


    // ============================================================
    // Ordered retirement FIFO
    //
    // 0 = Bank A
    // 1 = Bank B
    //
    // Depth is exactly two because at most two anchors can be
    // outstanding.
    // ============================================================

    logic order_fifo [0:1];

    logic order_rd_ptr;
    logic order_wr_ptr;

    logic [1:0] order_count;


    wire order_head =
        order_fifo[
            order_rd_ptr
        ];


    // ============================================================
    // BANK A side vectors
    // ============================================================

    logic [Q_BITS-1:0]
        a_q_l [0:DFL_BINS-1];

    logic [Q_BITS-1:0]
        a_q_t [0:DFL_BINS-1];

    logic [Q_BITS-1:0]
        a_q_r [0:DFL_BINS-1];

    logic [Q_BITS-1:0]
        a_q_b [0:DFL_BINS-1];


    // ============================================================
    // BANK B side vectors
    // ============================================================

    logic [Q_BITS-1:0]
        b_q_l [0:DFL_BINS-1];

    logic [Q_BITS-1:0]
        b_q_t [0:DFL_BINS-1];

    logic [Q_BITS-1:0]
        b_q_r [0:DFL_BINS-1];

    logic [Q_BITS-1:0]
        b_q_b [0:DFL_BINS-1];


    integer split_i;

    always_comb begin

        for (
            split_i = 0;
            split_i < DFL_BINS;
            split_i = split_i + 1
        ) begin

            // ====================================================
            // BANK A
            // ====================================================

            a_q_l[split_i] =
                bank_a_bbox[
                    split_i
                ];

            a_q_t[split_i] =
                bank_a_bbox[
                    DFL_BINS +
                    split_i
                ];

            a_q_r[split_i] =
                bank_a_bbox[
                    2*DFL_BINS +
                    split_i
                ];

            a_q_b[split_i] =
                bank_a_bbox[
                    3*DFL_BINS +
                    split_i
                ];


            // ====================================================
            // BANK B
            // ====================================================

            b_q_l[split_i] =
                bank_b_bbox[
                    split_i
                ];

            b_q_t[split_i] =
                bank_b_bbox[
                    DFL_BINS +
                    split_i
                ];

            b_q_r[split_i] =
                bank_b_bbox[
                    2*DFL_BINS +
                    split_i
                ];

            b_q_b[split_i] =
                bank_b_bbox[
                    3*DFL_BINS +
                    split_i
                ];

        end

    end


    // ============================================================
    // BANK A core interface
    // ============================================================

    logic a_l_ready;
    logic a_t_ready;
    logic a_r_ready;
    logic a_b_ready;

    logic a_core_start;


    logic a_l_out_valid;
    logic a_t_out_valid;
    logic a_r_out_valid;
    logic a_b_out_valid;


    logic [OFF_BITS-1:0]
        a_l_offset;

    logic [OFF_BITS-1:0]
        a_t_offset;

    logic [OFF_BITS-1:0]
        a_r_offset;

    logic [OFF_BITS-1:0]
        a_b_offset;


    logic a_done_l;
    logic a_done_t;
    logic a_done_r;
    logic a_done_b;


    assign a_core_start =
        bank_a_start_pending &&
        a_l_ready &&
        a_t_ready &&
        a_r_ready &&
        a_b_ready;


    // ============================================================
    // BANK B core interface
    // ============================================================

    logic b_l_ready;
    logic b_t_ready;
    logic b_r_ready;
    logic b_b_ready;

    logic b_core_start;


    logic b_l_out_valid;
    logic b_t_out_valid;
    logic b_r_out_valid;
    logic b_b_out_valid;


    logic [OFF_BITS-1:0]
        b_l_offset;

    logic [OFF_BITS-1:0]
        b_t_offset;

    logic [OFF_BITS-1:0]
        b_r_offset;

    logic [OFF_BITS-1:0]
        b_b_offset;


    logic b_done_l;
    logic b_done_t;
    logic b_done_r;
    logic b_done_b;


    assign b_core_start =
        bank_b_start_pending &&
        b_l_ready &&
        b_t_ready &&
        b_r_ready &&
        b_b_ready;


    // ============================================================
    // BANK A - LEFT
    // ============================================================

    dfl_core #(
        .Q_BITS(Q_BITS),
        .DFL_BINS(DFL_BINS),
        .LUT_WIDTH(LUT_WIDTH),
        .LUT_DEPTH(LUT_DEPTH),
        .LUT_FILE(LUT_FILE),
        .OFF_BITS(OFF_BITS),
        .OFF_FRAC(OFF_FRAC)
    ) u_bank_a_left (
        .clk(clk),
        .rst_n(rst_n),

        .in_valid(a_core_start),
        .in_ready(a_l_ready),

        .q(a_q_l),

        .out_valid(a_l_out_valid),
        .offset(a_l_offset)
    );


    // ============================================================
    // BANK A - TOP
    // ============================================================

    dfl_core #(
        .Q_BITS(Q_BITS),
        .DFL_BINS(DFL_BINS),
        .LUT_WIDTH(LUT_WIDTH),
        .LUT_DEPTH(LUT_DEPTH),
        .LUT_FILE(LUT_FILE),
        .OFF_BITS(OFF_BITS),
        .OFF_FRAC(OFF_FRAC)
    ) u_bank_a_top (
        .clk(clk),
        .rst_n(rst_n),

        .in_valid(a_core_start),
        .in_ready(a_t_ready),

        .q(a_q_t),

        .out_valid(a_t_out_valid),
        .offset(a_t_offset)
    );


    // ============================================================
    // BANK A - RIGHT
    // ============================================================

    dfl_core #(
        .Q_BITS(Q_BITS),
        .DFL_BINS(DFL_BINS),
        .LUT_WIDTH(LUT_WIDTH),
        .LUT_DEPTH(LUT_DEPTH),
        .LUT_FILE(LUT_FILE),
        .OFF_BITS(OFF_BITS),
        .OFF_FRAC(OFF_FRAC)
    ) u_bank_a_right (
        .clk(clk),
        .rst_n(rst_n),

        .in_valid(a_core_start),
        .in_ready(a_r_ready),

        .q(a_q_r),

        .out_valid(a_r_out_valid),
        .offset(a_r_offset)
    );


    // ============================================================
    // BANK A - BOTTOM
    // ============================================================

    dfl_core #(
        .Q_BITS(Q_BITS),
        .DFL_BINS(DFL_BINS),
        .LUT_WIDTH(LUT_WIDTH),
        .LUT_DEPTH(LUT_DEPTH),
        .LUT_FILE(LUT_FILE),
        .OFF_BITS(OFF_BITS),
        .OFF_FRAC(OFF_FRAC)
    ) u_bank_a_bottom (
        .clk(clk),
        .rst_n(rst_n),

        .in_valid(a_core_start),
        .in_ready(a_b_ready),

        .q(a_q_b),

        .out_valid(a_b_out_valid),
        .offset(a_b_offset)
    );


    // ============================================================
    // BANK B - LEFT
    // ============================================================

    dfl_core #(
        .Q_BITS(Q_BITS),
        .DFL_BINS(DFL_BINS),
        .LUT_WIDTH(LUT_WIDTH),
        .LUT_DEPTH(LUT_DEPTH),
        .LUT_FILE(LUT_FILE),
        .OFF_BITS(OFF_BITS),
        .OFF_FRAC(OFF_FRAC)
    ) u_bank_b_left (
        .clk(clk),
        .rst_n(rst_n),

        .in_valid(b_core_start),
        .in_ready(b_l_ready),

        .q(b_q_l),

        .out_valid(b_l_out_valid),
        .offset(b_l_offset)
    );


    // ============================================================
    // BANK B - TOP
    // ============================================================

    dfl_core #(
        .Q_BITS(Q_BITS),
        .DFL_BINS(DFL_BINS),
        .LUT_WIDTH(LUT_WIDTH),
        .LUT_DEPTH(LUT_DEPTH),
        .LUT_FILE(LUT_FILE),
        .OFF_BITS(OFF_BITS),
        .OFF_FRAC(OFF_FRAC)
    ) u_bank_b_top (
        .clk(clk),
        .rst_n(rst_n),

        .in_valid(b_core_start),
        .in_ready(b_t_ready),

        .q(b_q_t),

        .out_valid(b_t_out_valid),
        .offset(b_t_offset)
    );


    // ============================================================
    // BANK B - RIGHT
    // ============================================================

    dfl_core #(
        .Q_BITS(Q_BITS),
        .DFL_BINS(DFL_BINS),
        .LUT_WIDTH(LUT_WIDTH),
        .LUT_DEPTH(LUT_DEPTH),
        .LUT_FILE(LUT_FILE),
        .OFF_BITS(OFF_BITS),
        .OFF_FRAC(OFF_FRAC)
    ) u_bank_b_right (
        .clk(clk),
        .rst_n(rst_n),

        .in_valid(b_core_start),
        .in_ready(b_r_ready),

        .q(b_q_r),

        .out_valid(b_r_out_valid),
        .offset(b_r_offset)
    );


    // ============================================================
    // BANK B - BOTTOM
    // ============================================================

    dfl_core #(
        .Q_BITS(Q_BITS),
        .DFL_BINS(DFL_BINS),
        .LUT_WIDTH(LUT_WIDTH),
        .LUT_DEPTH(LUT_DEPTH),
        .LUT_FILE(LUT_FILE),
        .OFF_BITS(OFF_BITS),
        .OFF_FRAC(OFF_FRAC)
    ) u_bank_b_bottom (
        .clk(clk),
        .rst_n(rst_n),

        .in_valid(b_core_start),
        .in_ready(b_b_ready),

        .q(b_q_b),

        .out_valid(b_b_out_valid),
        .offset(b_b_offset)
    );


    // ============================================================
    // Retirement conditions
    //
    // A bank may retire only when:
    //
    //   1. it is the oldest outstanding transaction, and
    //   2. all four DFL side results are available.
    // ============================================================

    wire retire_a =
        (order_count != 0) &&
        (order_head == 1'b0) &&
        bank_a_result_ready;


    wire retire_b =
        (order_count != 0) &&
        (order_head == 1'b1) &&
        bank_b_result_ready;


    wire retire_any =
        retire_a ||
        retire_b;


    // ============================================================
    // Main sequential control
    // ============================================================

    integer bbox_i;


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            // ====================================================
            // Bank A
            // ====================================================

            bank_a_busy <=
                1'b0;

            bank_a_start_pending <=
                1'b0;

            bank_a_result_ready <=
                1'b0;


            bank_a_off_l <=
                '0;

            bank_a_off_t <=
                '0;

            bank_a_off_r <=
                '0;

            bank_a_off_b <=
                '0;


            a_done_l <=
                1'b0;

            a_done_t <=
                1'b0;

            a_done_r <=
                1'b0;

            a_done_b <=
                1'b0;


            // ====================================================
            // Bank B
            // ====================================================

            bank_b_busy <=
                1'b0;

            bank_b_start_pending <=
                1'b0;

            bank_b_result_ready <=
                1'b0;


            bank_b_off_l <=
                '0;

            bank_b_off_t <=
                '0;

            bank_b_off_r <=
                '0;

            bank_b_off_b <=
                '0;


            b_done_l <=
                1'b0;

            b_done_t <=
                1'b0;

            b_done_r <=
                1'b0;

            b_done_b <=
                1'b0;


            // ====================================================
            // Ordered FIFO
            // ====================================================

            order_fifo[0] <=
                1'b0;

            order_fifo[1] <=
                1'b0;

            order_rd_ptr <=
                1'b0;

            order_wr_ptr <=
                1'b0;

            order_count <=
                2'd0;


            // ====================================================
            // Output
            // ====================================================

            out_valid <=
                1'b0;

            off_l <=
                '0;

            off_t <=
                '0;

            off_r <=
                '0;

            off_b <=
                '0;


            // ====================================================
            // Input memories
            // ====================================================

            for (
                bbox_i = 0;
                bbox_i < TOTAL_BINS;
                bbox_i = bbox_i + 1
            ) begin

                bank_a_bbox[bbox_i] <=
                    '0;

                bank_b_bbox[bbox_i] <=
                    '0;

            end

        end


        else begin

            // ====================================================
            // Default output pulse
            // ====================================================

            out_valid <=
                1'b0;


            // ====================================================
            // Start Bank A
            // ====================================================

            if (a_core_start) begin

                bank_a_start_pending <=
                    1'b0;


                a_done_l <=
                    1'b0;

                a_done_t <=
                    1'b0;

                a_done_r <=
                    1'b0;

                a_done_b <=
                    1'b0;

            end


            // ====================================================
            // Start Bank B
            // ====================================================

            if (b_core_start) begin

                bank_b_start_pending <=
                    1'b0;


                b_done_l <=
                    1'b0;

                b_done_t <=
                    1'b0;

                b_done_r <=
                    1'b0;

                b_done_b <=
                    1'b0;

            end


            // ====================================================
            // Capture Bank A core outputs
            // ====================================================

            if (a_l_out_valid) begin

                bank_a_off_l <=
                    a_l_offset;

                a_done_l <=
                    1'b1;

            end


            if (a_t_out_valid) begin

                bank_a_off_t <=
                    a_t_offset;

                a_done_t <=
                    1'b1;

            end


            if (a_r_out_valid) begin

                bank_a_off_r <=
                    a_r_offset;

                a_done_r <=
                    1'b1;

            end


            if (a_b_out_valid) begin

                bank_a_off_b <=
                    a_b_offset;

                a_done_b <=
                    1'b1;

            end


            // ====================================================
            // Bank A result complete
            // ====================================================

            if (
                bank_a_busy &&
                !bank_a_result_ready &&
                (
                    a_done_l ||
                    a_l_out_valid
                )
                &&
                (
                    a_done_t ||
                    a_t_out_valid
                )
                &&
                (
                    a_done_r ||
                    a_r_out_valid
                )
                &&
                (
                    a_done_b ||
                    a_b_out_valid
                )
            ) begin

                bank_a_result_ready <=
                    1'b1;

            end


            // ====================================================
            // Capture Bank B core outputs
            // ====================================================

            if (b_l_out_valid) begin

                bank_b_off_l <=
                    b_l_offset;

                b_done_l <=
                    1'b1;

            end


            if (b_t_out_valid) begin

                bank_b_off_t <=
                    b_t_offset;

                b_done_t <=
                    1'b1;

            end


            if (b_r_out_valid) begin

                bank_b_off_r <=
                    b_r_offset;

                b_done_r <=
                    1'b1;

            end


            if (b_b_out_valid) begin

                bank_b_off_b <=
                    b_b_offset;

                b_done_b <=
                    1'b1;

            end


            // ====================================================
            // Bank B result complete
            // ====================================================

            if (
                bank_b_busy &&
                !bank_b_result_ready &&
                (
                    b_done_l ||
                    b_l_out_valid
                )
                &&
                (
                    b_done_t ||
                    b_t_out_valid
                )
                &&
                (
                    b_done_r ||
                    b_r_out_valid
                )
                &&
                (
                    b_done_b ||
                    b_b_out_valid
                )
            ) begin

                bank_b_result_ready <=
                    1'b1;

            end


            // ====================================================
            // Ordered retirement
            // ====================================================

            if (retire_a) begin

                off_l <=
                    bank_a_off_l;

                off_t <=
                    bank_a_off_t;

                off_r <=
                    bank_a_off_r;

                off_b <=
                    bank_a_off_b;


                out_valid <=
                    1'b1;


                bank_a_busy <=
                    1'b0;

                bank_a_result_ready <=
                    1'b0;


                a_done_l <=
                    1'b0;

                a_done_t <=
                    1'b0;

                a_done_r <=
                    1'b0;

                a_done_b <=
                    1'b0;

            end


            else if (retire_b) begin

                off_l <=
                    bank_b_off_l;

                off_t <=
                    bank_b_off_t;

                off_r <=
                    bank_b_off_r;

                off_b <=
                    bank_b_off_b;


                out_valid <=
                    1'b1;


                bank_b_busy <=
                    1'b0;

                bank_b_result_ready <=
                    1'b0;


                b_done_l <=
                    1'b0;

                b_done_t <=
                    1'b0;

                b_done_r <=
                    1'b0;

                b_done_b <=
                    1'b0;

            end


            // ====================================================
            // Accept new anchor into a free bank
            // ====================================================

            if (input_fire) begin

                // ------------------------------------------------
                // BANK A
                // ------------------------------------------------

                if (select_bank_a) begin

                    for (
                        bbox_i = 0;
                        bbox_i < TOTAL_BINS;
                        bbox_i = bbox_i + 1
                    ) begin

                        bank_a_bbox[bbox_i] <=
                            bbox_q[bbox_i];

                    end


                    bank_a_busy <=
                        1'b1;

                    bank_a_start_pending <=
                        1'b1;

                    bank_a_result_ready <=
                        1'b0;


                    a_done_l <=
                        1'b0;

                    a_done_t <=
                        1'b0;

                    a_done_r <=
                        1'b0;

                    a_done_b <=
                        1'b0;


                    // --------------------------------------------
                    // Push Bank A into order FIFO
                    // --------------------------------------------

                    order_fifo[
                        order_wr_ptr
                    ] <=
                        1'b0;

                end


                // ------------------------------------------------
                // BANK B
                // ------------------------------------------------

                else begin

                    for (
                        bbox_i = 0;
                        bbox_i < TOTAL_BINS;
                        bbox_i = bbox_i + 1
                    ) begin

                        bank_b_bbox[bbox_i] <=
                            bbox_q[bbox_i];

                    end


                    bank_b_busy <=
                        1'b1;

                    bank_b_start_pending <=
                        1'b1;

                    bank_b_result_ready <=
                        1'b0;


                    b_done_l <=
                        1'b0;

                    b_done_t <=
                        1'b0;

                    b_done_r <=
                        1'b0;

                    b_done_b <=
                        1'b0;


                    // --------------------------------------------
                    // Push Bank B into order FIFO
                    // --------------------------------------------

                    order_fifo[
                        order_wr_ptr
                    ] <=
                        1'b1;

                end

            end


            // ====================================================
            // FIFO pointer/count update
            //
            // Handle push and pop in the same clock correctly.
            // ====================================================

            case ({
                input_fire,
                retire_any
            })

                2'b10:
                begin

                    order_wr_ptr <=
                        order_wr_ptr + 1'b1;

                    order_count <=
                        order_count + 1'b1;

                end


                2'b01:
                begin

                    order_rd_ptr <=
                        order_rd_ptr + 1'b1;

                    order_count <=
                        order_count - 1'b1;

                end


                2'b11:
                begin

                    order_wr_ptr <=
                        order_wr_ptr + 1'b1;

                    order_rd_ptr <=
                        order_rd_ptr + 1'b1;

                    order_count <=
                        order_count;

                end


                default:
                begin

                    order_count <=
                        order_count;

                end

            endcase

        end

    end


    // ============================================================
    // Simulation checks
    // ============================================================

`ifndef SYNTHESIS

    initial begin

        if (DFL_BINS != 16) begin

            $error(
                "yolov8_dfl_top V5 requires DFL_BINS=16"
            );

        end


        if (DFL_SIDES != 4) begin

            $error(
                "yolov8_dfl_top V5 requires DFL_SIDES=4"
            );

        end

    end


    // ============================================================
    // Order FIFO may never exceed two outstanding anchors.
    // ============================================================

    always_ff @(posedge clk) begin

        if (rst_n) begin

            if (order_count > 2) begin

                $error(
                    "yolov8_dfl_top V5 order FIFO overflow"
                );

            end

        end

    end

`endif


endmodule