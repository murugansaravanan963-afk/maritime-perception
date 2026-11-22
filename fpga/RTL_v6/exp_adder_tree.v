`timescale 1ns / 1ps

module exp_adder_tree #(
    // ------------------------------------------------------------
    // DFL exp() LUT output width
    //
    // Current design:
    // LUT_WIDTH = 16
    // ------------------------------------------------------------
    parameter integer LUT_WIDTH = 16,

    // ------------------------------------------------------------
    // DFL structure
    //
    // Current YOLOv8 DFL uses 16 bins.
    //
    // This module's physical tree is still a 16-bin tree.
    // We keep DFL_BINS explicit so width derivation is documented.
    // ------------------------------------------------------------
    parameter integer DFL_BINS = 16,

    // ------------------------------------------------------------
    // Derived final widths
    //
    // sum_exp:
    //   sum of 16 LUT values
    //
    // weighted_sum:
    //   sum(i * e[i]), i = 0...15
    //
    // For current defaults:
    //
    // SUM_WIDTH          = 16 + log2(16) = 20
    //
    // WEIGHTED_SUM_WIDTH = 16 + 4 + 3 = 23
    //
    // The weighted width intentionally remains 23 bits to
    // preserve the already verified RTL numerical behavior.
    // ------------------------------------------------------------
    parameter integer SUM_WIDTH =
        LUT_WIDTH + $clog2(DFL_BINS),

    parameter integer INDEX_BITS =
        $clog2(DFL_BINS),

    parameter integer WEIGHTED_SUM_WIDTH =
        LUT_WIDTH + INDEX_BITS + 3
)(
    input  wire clk,
    input  wire rst_n,
    input  wire in_valid,

    // ============================================================
    // 16 exponential LUT values
    // ============================================================

    input wire [LUT_WIDTH-1:0] e0,
    input wire [LUT_WIDTH-1:0] e1,
    input wire [LUT_WIDTH-1:0] e2,
    input wire [LUT_WIDTH-1:0] e3,
    input wire [LUT_WIDTH-1:0] e4,
    input wire [LUT_WIDTH-1:0] e5,
    input wire [LUT_WIDTH-1:0] e6,
    input wire [LUT_WIDTH-1:0] e7,
    input wire [LUT_WIDTH-1:0] e8,
    input wire [LUT_WIDTH-1:0] e9,
    input wire [LUT_WIDTH-1:0] e10,
    input wire [LUT_WIDTH-1:0] e11,
    input wire [LUT_WIDTH-1:0] e12,
    input wire [LUT_WIDTH-1:0] e13,
    input wire [LUT_WIDTH-1:0] e14,
    input wire [LUT_WIDTH-1:0] e15,

    output reg out_valid,

    // S = sum(e[i])
    output reg [SUM_WIDTH-1:0] sum_exp,

    // N = sum(i * e[i])
    output reg [WEIGHTED_SUM_WIDTH-1:0] weighted_sum
);

    // ============================================================
    // Internal stage widths
    //
    // Normal sum path:
    //
    // 16 LUT values
    //   ↓
    // 8 pair sums   : LUT_WIDTH + 1
    //   ↓
    // 4 sums        : LUT_WIDTH + 2
    //   ↓
    // 2 sums        : LUT_WIDTH + 3
    //   ↓
    // final         : LUT_WIDTH + 4
    //
    // Weighted path preserves the exact widths of the verified
    // baseline:
    //
    // current defaults:
    //   Stage 1 = 21 bits
    //   Stage 2 = 22 bits
    //   Stage 3 = 23 bits
    //   Final   = 23 bits
    // ============================================================

    localparam integer S1_SUM_WIDTH =
        LUT_WIDTH + 1;

    localparam integer S2_SUM_WIDTH =
        LUT_WIDTH + 2;

    localparam integer S3_SUM_WIDTH =
        LUT_WIDTH + 3;


    // Weighted terms need enough width for multiplication by
    // bin index 0...15.
    //
    // Current:
    // 16 + 4 + 1 = 21 bits.

    localparam integer S1_WEIGHT_WIDTH =
        LUT_WIDTH + INDEX_BITS + 1;

    localparam integer S2_WEIGHT_WIDTH =
        S1_WEIGHT_WIDTH + 1;

    localparam integer S3_WEIGHT_WIDTH =
        WEIGHTED_SUM_WIDTH;


    // ============================================================
    // Stage 1
    // ============================================================

    reg [S1_SUM_WIDTH-1:0]
        s1 [0:7];

    reg [S1_WEIGHT_WIDTH-1:0]
        n1 [0:7];

    reg valid_s1;


    // ============================================================
    // Stage 2
    // ============================================================

    reg [S2_SUM_WIDTH-1:0]
        s2 [0:3];

    reg [S2_WEIGHT_WIDTH-1:0]
        n2 [0:3];

    reg valid_s2;


    // ============================================================
    // Stage 3
    // ============================================================

    reg [S3_SUM_WIDTH-1:0]
        s3 [0:1];

    reg [S3_WEIGHT_WIDTH-1:0]
        n3 [0:1];

    reg valid_s3;


    integer i;


    // ============================================================
    // Explicitly widened weighted operands
    //
    // Original verified RTL used:
    //
    // {5'b0, eX}
    //
    // for a 16-bit LUT value, producing 21 bits.
    //
    // We preserve exactly that behavior, but derive the extension
    // width from the parameters instead of hard-coding "5".
    // ============================================================

    wire [S1_WEIGHT_WIDTH-1:0] we0  =
        {{(S1_WEIGHT_WIDTH-LUT_WIDTH){1'b0}}, e0};

    wire [S1_WEIGHT_WIDTH-1:0] we1  =
        {{(S1_WEIGHT_WIDTH-LUT_WIDTH){1'b0}}, e1};

    wire [S1_WEIGHT_WIDTH-1:0] we2  =
        {{(S1_WEIGHT_WIDTH-LUT_WIDTH){1'b0}}, e2};

    wire [S1_WEIGHT_WIDTH-1:0] we3  =
        {{(S1_WEIGHT_WIDTH-LUT_WIDTH){1'b0}}, e3};

    wire [S1_WEIGHT_WIDTH-1:0] we4  =
        {{(S1_WEIGHT_WIDTH-LUT_WIDTH){1'b0}}, e4};

    wire [S1_WEIGHT_WIDTH-1:0] we5  =
        {{(S1_WEIGHT_WIDTH-LUT_WIDTH){1'b0}}, e5};

    wire [S1_WEIGHT_WIDTH-1:0] we6  =
        {{(S1_WEIGHT_WIDTH-LUT_WIDTH){1'b0}}, e6};

    wire [S1_WEIGHT_WIDTH-1:0] we7  =
        {{(S1_WEIGHT_WIDTH-LUT_WIDTH){1'b0}}, e7};

    wire [S1_WEIGHT_WIDTH-1:0] we8  =
        {{(S1_WEIGHT_WIDTH-LUT_WIDTH){1'b0}}, e8};

    wire [S1_WEIGHT_WIDTH-1:0] we9  =
        {{(S1_WEIGHT_WIDTH-LUT_WIDTH){1'b0}}, e9};

    wire [S1_WEIGHT_WIDTH-1:0] we10 =
        {{(S1_WEIGHT_WIDTH-LUT_WIDTH){1'b0}}, e10};

    wire [S1_WEIGHT_WIDTH-1:0] we11 =
        {{(S1_WEIGHT_WIDTH-LUT_WIDTH){1'b0}}, e11};

    wire [S1_WEIGHT_WIDTH-1:0] we12 =
        {{(S1_WEIGHT_WIDTH-LUT_WIDTH){1'b0}}, e12};

    wire [S1_WEIGHT_WIDTH-1:0] we13 =
        {{(S1_WEIGHT_WIDTH-LUT_WIDTH){1'b0}}, e13};

    wire [S1_WEIGHT_WIDTH-1:0] we14 =
        {{(S1_WEIGHT_WIDTH-LUT_WIDTH){1'b0}}, e14};

    wire [S1_WEIGHT_WIDTH-1:0] we15 =
        {{(S1_WEIGHT_WIDTH-LUT_WIDTH){1'b0}}, e15};


    // ============================================================
    // Stage 1
    //
    // S pair sums
    //
    // Weighted sum:
    //
    // N = 0*e0 + 1*e1 + ... + 15*e15
    //
    // Constant multiplication remains shifts/adds exactly as in
    // the verified baseline.
    // ============================================================

    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            valid_s1 <= 1'b0;

            for (
                i = 0;
                i < 8;
                i = i + 1
            ) begin

                s1[i] <= 0;
                n1[i] <= 0;

            end

        end
        else begin

            valid_s1 <= in_valid;


            if (in_valid) begin

                // ------------------------------------------------
                // Normal exponential sum
                // ------------------------------------------------

                s1[0] <= e0  + e1;
                s1[1] <= e2  + e3;
                s1[2] <= e4  + e5;
                s1[3] <= e6  + e7;

                s1[4] <= e8  + e9;
                s1[5] <= e10 + e11;
                s1[6] <= e12 + e13;
                s1[7] <= e14 + e15;


                // ------------------------------------------------
                // Weighted exponential sum
                //
                // pair 0:
                // 0*e0 + 1*e1
                // ------------------------------------------------

                n1[0] <=
                    we1;


                // ------------------------------------------------
                // pair 1:
                // 2*e2 + 3*e3
                // ------------------------------------------------

                n1[1] <=
                    (we2 << 1)
                    +
                    (
                        (we3 << 1)
                        +
                        we3
                    );


                // ------------------------------------------------
                // pair 2:
                // 4*e4 + 5*e5
                // ------------------------------------------------

                n1[2] <=
                    (we4 << 2)
                    +
                    (
                        (we5 << 2)
                        +
                        we5
                    );


                // ------------------------------------------------
                // pair 3:
                // 6*e6 + 7*e7
                // ------------------------------------------------

                n1[3] <=
                    (
                        (we6 << 2)
                        +
                        (we6 << 1)
                    )
                    +
                    (
                        (we7 << 3)
                        -
                        we7
                    );


                // ------------------------------------------------
                // pair 4:
                // 8*e8 + 9*e9
                // ------------------------------------------------

                n1[4] <=
                    (we8 << 3)
                    +
                    (
                        (we9 << 3)
                        +
                        we9
                    );


                // ------------------------------------------------
                // pair 5:
                // 10*e10 + 11*e11
                // ------------------------------------------------

                n1[5] <=
                    (
                        (we10 << 3)
                        +
                        (we10 << 1)
                    )
                    +
                    (
                        (we11 << 3)
                        +
                        (we11 << 1)
                        +
                        we11
                    );


                // ------------------------------------------------
                // pair 6:
                // 12*e12 + 13*e13
                // ------------------------------------------------

                n1[6] <=
                    (
                        (we12 << 3)
                        +
                        (we12 << 2)
                    )
                    +
                    (
                        (we13 << 3)
                        +
                        (we13 << 2)
                        +
                        we13
                    );


                // ------------------------------------------------
                // pair 7:
                // 14*e14 + 15*e15
                // ------------------------------------------------

                n1[7] <=
                    (
                        (we14 << 4)
                        -
                        (we14 << 1)
                    )
                    +
                    (
                        (we15 << 4)
                        -
                        we15
                    );

            end

        end

    end


    // ============================================================
    // Stage 2
    // ============================================================

    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            valid_s2 <= 1'b0;

            for (
                i = 0;
                i < 4;
                i = i + 1
            ) begin

                s2[i] <= 0;
                n2[i] <= 0;

            end

        end
        else begin

            valid_s2 <=
                valid_s1;


            if (valid_s1) begin

                s2[0] <=
                    s1[0] + s1[1];

                s2[1] <=
                    s1[2] + s1[3];

                s2[2] <=
                    s1[4] + s1[5];

                s2[3] <=
                    s1[6] + s1[7];


                n2[0] <=
                    n1[0] + n1[1];

                n2[1] <=
                    n1[2] + n1[3];

                n2[2] <=
                    n1[4] + n1[5];

                n2[3] <=
                    n1[6] + n1[7];

            end

        end

    end


    // ============================================================
    // Stage 3
    // ============================================================

    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            valid_s3 <=
                1'b0;

            s3[0] <= 0;
            s3[1] <= 0;

            n3[0] <= 0;
            n3[1] <= 0;

        end
        else begin

            valid_s3 <=
                valid_s2;


            if (valid_s2) begin

                s3[0] <=
                    s2[0] + s2[1];

                s3[1] <=
                    s2[2] + s2[3];


                n3[0] <=
                    n2[0] + n2[1];

                n3[1] <=
                    n2[2] + n2[3];

            end

        end

    end


    // ============================================================
    // Stage 4
    //
    // Final output widths are parameterized, but with current
    // defaults remain exactly:
    //
    // sum_exp      = 20 bits
    // weighted_sum = 23 bits
    // ============================================================

    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            out_valid <=
                1'b0;

            sum_exp <=
                {SUM_WIDTH{1'b0}};

            weighted_sum <=
                {WEIGHTED_SUM_WIDTH{1'b0}};

        end
        else begin

            out_valid <=
                valid_s3;


            if (valid_s3) begin

                sum_exp <=
                    s3[0] + s3[1];

                weighted_sum <=
                    n3[0] + n3[1];

            end

        end

    end


    // ============================================================
    // Simulation-only configuration checks
    // ============================================================

`ifndef SYNTHESIS

    initial begin

        // Physical shift/add tree is currently the YOLOv8
        // 16-bin DFL implementation.
        if (DFL_BINS != 16) begin

            $error(
                "exp_adder_tree: current physical tree requires DFL_BINS=16"
            );

        end


        if (LUT_WIDTH < 1) begin

            $error(
                "exp_adder_tree: LUT_WIDTH must be >= 1"
            );

        end

    end

`endif

endmodule