`timescale 1ns / 1ps

module dfl_core #(
    // ============================================================
    // Quantized DFL input
    // ============================================================

    parameter integer Q_BITS = 8,

    // ============================================================
    // YOLOv8 DFL structure
    // ============================================================

    parameter integer DFL_BINS = 16,

    // ============================================================
    // Exponential LUT
    // ============================================================

    parameter integer LUT_WIDTH = 16,
    parameter integer LUT_DEPTH = 65,

    parameter LUT_FILE =
        "dfl_exp_lut.mem",

    // ============================================================
    // DFL output fixed-point format
    //
    // Current:
    // Q4.8
    // ============================================================

    parameter integer OFF_BITS = 12,
    parameter integer OFF_FRAC = 8,

    // ============================================================
    // Derived widths
    // ============================================================

    parameter integer SUM_WIDTH =
        LUT_WIDTH + $clog2(DFL_BINS),

    parameter integer INDEX_BITS =
        $clog2(DFL_BINS),

    parameter integer WEIGHTED_SUM_WIDTH =
        LUT_WIDTH + INDEX_BITS + 3
)(
    input logic clk,
    input logic rst_n,

    // ============================================================
    // Input handshake
    //
    // A transaction is accepted only when:
    //
    //     in_valid && in_ready
    // ============================================================

    input logic in_valid,

    output logic in_ready,

    // ============================================================
    // One DFL side
    // ============================================================

    input logic [Q_BITS-1:0]
        q [0:DFL_BINS-1],

    // ============================================================
    // Output
    // ============================================================

    output logic out_valid,

    output logic [OFF_BITS-1:0]
        offset
);


    // ============================================================
    // Core transaction state
    //
    // IDLE:
    //     ready to accept one DFL side
    //
    // ACTIVE:
    //     one transaction is travelling through the pipeline /
    //     multi-cycle divider
    //
    // We intentionally allow only ONE outstanding transaction.
    // ============================================================

    logic core_busy;

    wire accept_input;


    assign accept_input =
        in_valid &&
        in_ready;


    always_comb begin

        in_ready =
            !core_busy;

    end


    // ============================================================
    // Input register
    //
    // The complete 16-bin input is captured only when the
    // handshake succeeds.
    // ============================================================

    logic [Q_BITS-1:0]
        q_reg [0:DFL_BINS-1];

    integer q_i;


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            for (
                q_i = 0;
                q_i < DFL_BINS;
                q_i = q_i + 1
            ) begin

                q_reg[q_i] <=
                    {Q_BITS{1'b0}};

            end

        end
        else begin

            if (accept_input) begin

                for (
                    q_i = 0;
                    q_i < DFL_BINS;
                    q_i = q_i + 1
                ) begin

                    q_reg[q_i] <=
                        q[q_i];

                end

            end

        end

    end


    // ============================================================
    // Start pulse for the internal pipeline
    //
    // q_reg is updated on the same edge as accept_input.
    //
    // Therefore the actual pipeline start must be delayed by
    // one clock so that downscale_block sees the newly registered
    // q_reg values.
    // ============================================================

    logic start_pipeline;


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            start_pipeline <=
                1'b0;

        end
        else begin

            start_pipeline <=
                accept_input;

        end

    end


    // ============================================================
    // Transaction busy control
    //
    // Set busy when a transaction is accepted.
    //
    // Clear busy when the final divider result is produced.
    // ============================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            core_busy <=
                1'b0;

        end
        else begin

            if (accept_input) begin

                core_busy <=
                    1'b1;

            end


            if (out_valid) begin

                core_busy <=
                    1'b0;

            end

        end

    end


    // ============================================================
    // Stage 1:
    // Downscale / max normalization
    // ============================================================

    logic downscale_valid;

    logic [Q_BITS-1:0]
        q_ds [0:DFL_BINS-1];


    downscale_block #(
        .Q_BITS(
            Q_BITS
        )
    ) u_downscale (
        .clk(
            clk
        ),

        .rst_n(
            rst_n
        ),

        .in_valid(
            start_pipeline
        ),

        .q0(
            q_reg[0]
        ),

        .q1(
            q_reg[1]
        ),

        .q2(
            q_reg[2]
        ),

        .q3(
            q_reg[3]
        ),

        .q4(
            q_reg[4]
        ),

        .q5(
            q_reg[5]
        ),

        .q6(
            q_reg[6]
        ),

        .q7(
            q_reg[7]
        ),

        .q8(
            q_reg[8]
        ),

        .q9(
            q_reg[9]
        ),

        .q10(
            q_reg[10]
        ),

        .q11(
            q_reg[11]
        ),

        .q12(
            q_reg[12]
        ),

        .q13(
            q_reg[13]
        ),

        .q14(
            q_reg[14]
        ),

        .q15(
            q_reg[15]
        ),

        .out_valid(
            downscale_valid
        ),

        .d0(
            q_ds[0]
        ),

        .d1(
            q_ds[1]
        ),

        .d2(
            q_ds[2]
        ),

        .d3(
            q_ds[3]
        ),

        .d4(
            q_ds[4]
        ),

        .d5(
            q_ds[5]
        ),

        .d6(
            q_ds[6]
        ),

        .d7(
            q_ds[7]
        ),

        .d8(
            q_ds[8]
        ),

        .d9(
            q_ds[9]
        ),

        .d10(
            q_ds[10]
        ),

        .d11(
            q_ds[11]
        ),

        .d12(
            q_ds[12]
        ),

        .d13(
            q_ds[13]
        ),

        .d14(
            q_ds[14]
        ),

        .d15(
            q_ds[15]
        )
    );


    // ============================================================
    // Stage 2A:
    // Combinational exponential LUT
    // ============================================================

    logic [LUT_WIDTH-1:0]
        exp_value_comb [0:DFL_BINS-1];


    genvar gi;


    generate

        for (
            gi = 0;
            gi < DFL_BINS;
            gi = gi + 1
        ) begin : GEN_EXP_LUT

            dfl_exp_lut #(
                .Q_BITS(
                    Q_BITS
                ),

                .LUT_WIDTH(
                    LUT_WIDTH
                ),

                .LUT_DEPTH(
                    LUT_DEPTH
                ),

                .LUT_FILE(
                    LUT_FILE
                )
            ) u_exp_lut (
                .d(
                    q_ds[gi]
                ),

                .exp_out(
                    exp_value_comb[gi]
                )
            );

        end

    endgenerate


    // ============================================================
    // Stage 2B:
    // Registered exponential outputs
    //
    // This pipeline stage was added for FPGA timing closure.
    // ============================================================

    logic exp_valid_reg;

    logic [LUT_WIDTH-1:0]
        exp_value_reg [0:DFL_BINS-1];

    integer exp_i;


    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            exp_valid_reg <=
                1'b0;


            for (
                exp_i = 0;
                exp_i < DFL_BINS;
                exp_i = exp_i + 1
            ) begin

                exp_value_reg[exp_i] <=
                    {LUT_WIDTH{1'b0}};

            end

        end
        else begin

            exp_valid_reg <=
                downscale_valid;


            if (downscale_valid) begin

                for (
                    exp_i = 0;
                    exp_i < DFL_BINS;
                    exp_i = exp_i + 1
                ) begin

                    exp_value_reg[exp_i] <=
                        exp_value_comb[exp_i];

                end

            end

        end

    end


    // ============================================================
    // Stage 3:
    // Exponential reduction / weighted-sum tree
    //
    // Computes:
    //
    //     sum_exp      = sum(exp_i)
    //
    //     weighted_sum = sum(i * exp_i)
    // ============================================================

    logic adder_valid;

    logic [SUM_WIDTH-1:0]
        sum_exp;

    logic [WEIGHTED_SUM_WIDTH-1:0]
        weighted_sum;


    exp_adder_tree #(
        .LUT_WIDTH(
            LUT_WIDTH
        ),

        .DFL_BINS(
            DFL_BINS
        ),

        .SUM_WIDTH(
            SUM_WIDTH
        ),

        .INDEX_BITS(
            INDEX_BITS
        ),

        .WEIGHTED_SUM_WIDTH(
            WEIGHTED_SUM_WIDTH
        )
    ) u_exp_adder_tree (
        .clk(
            clk
        ),

        .rst_n(
            rst_n
        ),

        .in_valid(
            exp_valid_reg
        ),

        .e0(
            exp_value_reg[0]
        ),

        .e1(
            exp_value_reg[1]
        ),

        .e2(
            exp_value_reg[2]
        ),

        .e3(
            exp_value_reg[3]
        ),

        .e4(
            exp_value_reg[4]
        ),

        .e5(
            exp_value_reg[5]
        ),

        .e6(
            exp_value_reg[6]
        ),

        .e7(
            exp_value_reg[7]
        ),

        .e8(
            exp_value_reg[8]
        ),

        .e9(
            exp_value_reg[9]
        ),

        .e10(
            exp_value_reg[10]
        ),

        .e11(
            exp_value_reg[11]
        ),

        .e12(
            exp_value_reg[12]
        ),

        .e13(
            exp_value_reg[13]
        ),

        .e14(
            exp_value_reg[14]
        ),

        .e15(
            exp_value_reg[15]
        ),

        .out_valid(
            adder_valid
        ),

        .sum_exp(
            sum_exp
        ),

        .weighted_sum(
            weighted_sum
        )
    );


    // ============================================================
    // Stage 4:
    // Multi-cycle restoring divider
    // ============================================================

    logic divider_busy;

    logic divider_out_valid;

    logic [OFF_BITS-1:0]
        divider_offset;


    dfl_divider #(
        .SUM_WIDTH(
            SUM_WIDTH
        ),

        .WEIGHTED_SUM_WIDTH(
            WEIGHTED_SUM_WIDTH
        ),

        .OFF_BITS(
            OFF_BITS
        ),

        .OFF_FRAC(
            OFF_FRAC
        ),

        .NUM_WIDTH(
            WEIGHTED_SUM_WIDTH +
            OFF_FRAC
        )
    ) u_divider (
        .clk(
            clk
        ),

        .rst_n(
            rst_n
        ),

        .in_valid(
            adder_valid
        ),

        .sum_exp(
            sum_exp
        ),

        .weighted_sum(
            weighted_sum
        ),

        .busy(
            divider_busy
        ),

        .out_valid(
            divider_out_valid
        ),

        .offset(
            divider_offset
        )
    );


    // ============================================================
    // Final output
    //
    // Divider out_valid is already a one-clock pulse.
    // ============================================================

    always_ff @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            out_valid <=
                1'b0;


            offset <=
                {OFF_BITS{1'b0}};

        end
        else begin

            out_valid <=
                divider_out_valid;


            if (divider_out_valid) begin

                offset <=
                    divider_offset;

            end

        end

    end


    // ============================================================
    // Simulation-only checks
    // ============================================================

`ifndef SYNTHESIS

    initial begin

        if (
            DFL_BINS != 16
        ) begin

            $error(
                "dfl_core: current physical implementation requires DFL_BINS=16"
            );

        end


        if (
            Q_BITS < 1
        ) begin

            $error(
                "dfl_core: Q_BITS must be >= 1"
            );

        end


        if (
            LUT_WIDTH < 1
        ) begin

            $error(
                "dfl_core: LUT_WIDTH must be >= 1"
            );

        end


        if (
            LUT_DEPTH < 1
        ) begin

            $error(
                "dfl_core: LUT_DEPTH must be >= 1"
            );

        end


        if (
            OFF_BITS < 1
        ) begin

            $error(
                "dfl_core: OFF_BITS must be >= 1"
            );

        end


        if (
            OFF_FRAC < 0
        ) begin

            $error(
                "dfl_core: OFF_FRAC must be >= 0"
            );

        end


        if (
            SUM_WIDTH <
            (
                LUT_WIDTH +
                $clog2(DFL_BINS)
            )
        ) begin

            $error(
                "dfl_core: SUM_WIDTH is too small"
            );

        end

    end


    // ------------------------------------------------------------
    // With the one-outstanding-transaction handshake,
    // adder_valid must never reach a busy divider.
    // ------------------------------------------------------------

    always_ff @(posedge clk) begin

        if (
            rst_n &&
            adder_valid &&
            divider_busy
        ) begin

            $error(
                "dfl_core: internal protocol error - divider busy when adder result arrived"
            );

        end

    end

`endif

endmodule