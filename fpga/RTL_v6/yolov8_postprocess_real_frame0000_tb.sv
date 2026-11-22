`timescale 1ns / 1ps

module yolov8_postprocess_real_frame0000_tb;

    // ============================================================
    // Model configuration
    //
    // Same auto-generated configuration file used by the DUT
    // wrapper.
    //
    // IMPORTANT:
    // These parameters are used ONLY by the testbench to size
    // memories / ports and load the corresponding vectors.
    //
    // The DUT itself obtains the same model configuration through
    // yolov8_postprocess_wrapper.
    // ============================================================

    `include "rtl_params.vh"


    // ============================================================
    // Architecture constants
    //
    // Current physical DFL architecture:
    //
    // 4 sides x DFL_BINS
    // ============================================================

    localparam integer DFL_SIDES = 4;

    localparam integer BBOX_PER_ANCHOR =
        DFL_SIDES * DFL_BINS;


    // Maritime960 golden checkpoint formats
    localparam integer DFL_OFF_BITS = 12;
    localparam integer DFL_GOLDEN_FIELDS = 4;
    localparam integer DECODE_GOLDEN_FIELDS = 7;

    localparam integer TOTAL_DFL_GOLDEN =
        N_ANCHORS * DFL_GOLDEN_FIELDS;

    localparam integer TOTAL_DECODE_GOLDEN =
        N_ANCHORS * DECODE_GOLDEN_FIELDS;


    // ============================================================
    // Input vector sizes
    // ============================================================

    localparam integer TOTAL_BBOX =
        N_ANCHORS * BBOX_PER_ANCHOR;

    localparam integer TOTAL_SCORE =
        N_ANCHORS * NUM_CLASSES;


    // ============================================================
    // Golden NMS format
    //
    // exp_nms.hex:
    //
    // first two values:
    //
    // n_det
    // overflow
    //
    // followed by 7 values per detection:
    //
    // x1
    // y1
    // x2
    // y2
    // cls
    // score
    // anchor
    // ============================================================

    localparam integer NMS_FIELDS_MAX =
        2 + MAX_DET * 7;


    // ============================================================
    // Clock / reset
    // ============================================================

    logic clk;
    logic rst_n;


    // ============================================================
    // Frame / input control
    // ============================================================

    logic frame_start;

    logic in_valid;
    logic in_ready;


    // ============================================================
    // Raw YOLO input
    // ============================================================

    logic [ANCHOR_BITS-1:0]
        anchor_idx;


    logic [Q_BITS-1:0]
        bbox_q [0:BBOX_PER_ANCHOR-1];


    logic [SCORE_Q_BITS-1:0]
        score_q [0:NUM_CLASSES-1];


    // ============================================================
    // Final RTL detection output
    // ============================================================

    logic det_valid;


    logic [COORD_BITS-1:0]
        det_x1;

    logic [COORD_BITS-1:0]
        det_y1;

    logic [COORD_BITS-1:0]
        det_x2;

    logic [COORD_BITS-1:0]
        det_y2;


    logic [CLASS_BITS-1:0]
        det_cls;


    logic [SCORE_Q_BITS-1:0]
        det_score;


    logic [ANCHOR_BITS-1:0]
        det_anchor_idx;


    logic [$clog2(MAX_DET+1)-1:0]
        n_det;


    logic overflow;

    logic done;


    // ============================================================
    // Input memories
    // ============================================================

    logic [Q_BITS-1:0]
        bbox_mem [0:TOTAL_BBOX-1];


    logic [SCORE_Q_BITS-1:0]
        score_mem [0:TOTAL_SCORE-1];


    // ============================================================
    // Intermediate golden checkpoints
    // exp_dfl.hex    : off_l off_t off_r off_b per anchor
    // exp_decode.hex : x1 y1 x2 y2 cls score pass per anchor
    // ============================================================

    logic [DFL_OFF_BITS-1:0]
        dfl_golden_mem [0:TOTAL_DFL_GOLDEN-1];

    logic [COORD_BITS-1:0]
        decode_golden_mem [0:TOTAL_DECODE_GOLDEN-1];


    // ============================================================
    // Golden NMS memory
    //
    // IMPORTANT:
    // Maritime960 uses COORD_BITS=14 but ANCHOR_BITS=15.
    // exp_nms.hex stores x/y/class/score/anchor in one flat memory,
    // so this memory must be wide enough for the largest field.
    // ANCHOR_BITS is currently the widest field (15 bits).
    // ============================================================

    logic [ANCHOR_BITS-1:0]
        nms_mem [0:NMS_FIELDS_MAX-1];


    // ============================================================
    // Golden header
    // ============================================================

    integer expected_n_det;

    integer expected_overflow;


    // ============================================================
    // Statistics
    // ============================================================

    integer input_anchor;

    integer output_idx;


    integer detections_checked;

    integer detections_matched;

    integer detections_mismatched;


    integer field_mismatches;

    integer mismatch_printed;

    integer local_errors;


    integer dfl_checked;
    integer dfl_mismatched;
    integer dfl_mismatch_printed;

    integer decode_checked;
    integer decode_mismatched;
    integer decode_mismatch_printed;


    // ============================================================
    // Timing statistics
    // ============================================================

    integer cycle_count;

    integer first_input_cycle;

    integer last_input_cycle;

    integer first_det_cycle;

    integer last_det_cycle;

    integer done_cycle;

    integer timeout_counter;


    integer i;


    // ============================================================
    // DUT
    //
    // IMPORTANT:
    //
    // No model parameters are manually passed here.
    //
    // yolov8_postprocess_wrapper obtains the model configuration
    // from rtl_params.vh.
    //
    // This is the final parameterized-wrapper regression path:
    //
    // rtl_params.vh
    //      ↓
    // wrapper
    //      ↓
    // complete post-processing RTL
    // ============================================================

    yolov8_postprocess_wrapper dut (
        .clk(
            clk
        ),

        .rst_n(
            rst_n
        ),

        .frame_start(
            frame_start
        ),

        .in_valid(
            in_valid
        ),

        .in_ready(
            in_ready
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
    // 100 MHz clock
    //
    // Period = 10 ns
    // ============================================================

    initial begin

        clk =
            1'b0;

        forever
            #5 clk = ~clk;

    end


    // ============================================================
    // Cycle counter
    // ============================================================

    always @(posedge clk or negedge rst_n) begin

        if (!rst_n) begin

            cycle_count <=
                0;

        end
        else begin

            cycle_count <=
                cycle_count + 1;

        end

    end


    // ============================================================
    // Load vectors
    //
    // IMPORTANT:
    //
    // This is still a true end-to-end test.
    //
    // RTL receives ONLY:
    //
    // in_bbox.hex
    // in_score.hex
    //
    // Final output is compared ONLY against:
    //
    // exp_nms.hex
    //
    // exp_dfl.hex / exp_decode.hex are NOT fed into the DUT.
    // ============================================================

    initial begin

        $display("");

        $display(
            "[TB] Loading raw bbox vectors..."
        );


        $readmemh(
            "frame_0000/in_bbox.hex",
            bbox_mem
        );


        $display(
            "[TB] Loading raw score vectors..."
        );


        $readmemh(
            "frame_0000/in_score.hex",
            score_mem
        );


        $display(
            "[TB] Loading DFL checkpoint..."
        );

        $readmemh(
            "frame_0000/exp_dfl.hex",
            dfl_golden_mem
        );

        $display(
            "[TB] Loading Decode checkpoint..."
        );

        $readmemh(
            "frame_0000/exp_decode.hex",
            decode_golden_mem
        );

        $display(
            "[TB] Loading final NMS golden output..."
        );


        $readmemh(
            "frame_0000/exp_nms.hex",
            nms_mem
        );


        #1;


        expected_n_det =
            nms_mem[0];


        expected_overflow =
            nms_mem[1][0];


        $display(
            "[TB] Parameterized-wrapper vectors loaded."
        );


        $display(
            "[TB] Golden n_det    = %0d",
            expected_n_det
        );


        $display(
            "[TB] Golden overflow = %0d",
            expected_overflow
        );

    end


    // ============================================================
    // Load one raw anchor into DUT input arrays
    // ============================================================

    task automatic load_input_anchor(
        input integer index
    );

        integer k;

        integer bbox_base;

        integer score_base;


        begin

            bbox_base =
                index *
                BBOX_PER_ANCHOR;


            score_base =
                index *
                NUM_CLASSES;


            // ----------------------------------------------------
            // BBox logits
            // ----------------------------------------------------

            for (
                k = 0;
                k < BBOX_PER_ANCHOR;
                k = k + 1
            ) begin

                bbox_q[k] =
                    bbox_mem[
                        bbox_base + k
                    ];

            end


            // ----------------------------------------------------
            // Quantized class scores
            // ----------------------------------------------------

            for (
                k = 0;
                k < NUM_CLASSES;
                k = k + 1
            ) begin

                score_q[k] =
                    score_mem[
                        score_base + k
                    ];

            end

        end

    endtask


    // ============================================================
    // Checkpoint 1: DFL exact checker
    //
    // The golden checkpoint is NOT fed into the DUT.  We only
    // observe the ordered DFL retirement interface inside decode.
    // ============================================================

    always @(posedge clk) begin
        integer base;
        integer errs;

        #1;

        if (!rst_n) begin
            dfl_checked = 0;
            dfl_mismatched = 0;
            dfl_mismatch_printed = 0;
        end
        else if (dut.u_postprocess.u_decode.dfl_valid) begin
            base = dfl_checked * DFL_GOLDEN_FIELDS;
            errs = 0;

            if (dut.u_postprocess.u_decode.off_l !== dfl_golden_mem[base+0]) errs = errs + 1;
            if (dut.u_postprocess.u_decode.off_t !== dfl_golden_mem[base+1]) errs = errs + 1;
            if (dut.u_postprocess.u_decode.off_r !== dfl_golden_mem[base+2]) errs = errs + 1;
            if (dut.u_postprocess.u_decode.off_b !== dfl_golden_mem[base+3]) errs = errs + 1;

            if (errs != 0) begin
                dfl_mismatched = dfl_mismatched + 1;
                if (dfl_mismatch_printed < 10) begin
                    $display("[DFL MISMATCH] anchor=%0d RTL=%03h %03h %03h %03h GOLDEN=%03h %03h %03h %03h",
                        dfl_checked,
                        dut.u_postprocess.u_decode.off_l,
                        dut.u_postprocess.u_decode.off_t,
                        dut.u_postprocess.u_decode.off_r,
                        dut.u_postprocess.u_decode.off_b,
                        dfl_golden_mem[base+0],
                        dfl_golden_mem[base+1],
                        dfl_golden_mem[base+2],
                        dfl_golden_mem[base+3]);
                    dfl_mismatch_printed = dfl_mismatch_printed + 1;
                end
            end

            dfl_checked = dfl_checked + 1;
        end
    end


    // ============================================================
    // Checkpoint 2: Decode exact checker
    //
    // exp_decode.hex fields per anchor:
    // x1 y1 x2 y2 cls score pass
    // ============================================================

    always @(posedge clk) begin
        integer base;
        integer errs;

        #1;

        if (!rst_n) begin
            decode_checked = 0;
            decode_mismatched = 0;
            decode_mismatch_printed = 0;
        end
        else if (dut.u_postprocess.decode_valid) begin
            base = decode_checked * DECODE_GOLDEN_FIELDS;
            errs = 0;

            if (dut.u_postprocess.decode_x1 !== decode_golden_mem[base+0]) errs = errs + 1;
            if (dut.u_postprocess.decode_y1 !== decode_golden_mem[base+1]) errs = errs + 1;
            if (dut.u_postprocess.decode_x2 !== decode_golden_mem[base+2]) errs = errs + 1;
            if (dut.u_postprocess.decode_y2 !== decode_golden_mem[base+3]) errs = errs + 1;
            if (dut.u_postprocess.decode_cls !== decode_golden_mem[base+4][CLASS_BITS-1:0]) errs = errs + 1;
            if (dut.u_postprocess.decode_score !== decode_golden_mem[base+5][SCORE_Q_BITS-1:0]) errs = errs + 1;
            if (dut.u_postprocess.decode_pass !== decode_golden_mem[base+6][0]) errs = errs + 1;

            if (errs != 0) begin
                decode_mismatched = decode_mismatched + 1;
                if (decode_mismatch_printed < 10) begin
                    $display("[DECODE MISMATCH] anchor=%0d", decode_checked);
                    $display("  RTL    box=(%0d,%0d)-(%0d,%0d) cls=%0d score=%0d pass=%0d",
                        dut.u_postprocess.decode_x1, dut.u_postprocess.decode_y1,
                        dut.u_postprocess.decode_x2, dut.u_postprocess.decode_y2,
                        dut.u_postprocess.decode_cls, dut.u_postprocess.decode_score,
                        dut.u_postprocess.decode_pass);
                    $display("  GOLDEN box=(%0d,%0d)-(%0d,%0d) cls=%0d score=%0d pass=%0d",
                        decode_golden_mem[base+0], decode_golden_mem[base+1],
                        decode_golden_mem[base+2], decode_golden_mem[base+3],
                        decode_golden_mem[base+4], decode_golden_mem[base+5],
                        decode_golden_mem[base+6]);
                    decode_mismatch_printed = decode_mismatch_printed + 1;
                end
            end

            decode_checked = decode_checked + 1;
        end
    end


    // ============================================================
    // Detection monitor
    // ============================================================

    always @(posedge clk) begin : detection_monitor

        integer golden_base;


        reg [COORD_BITS-1:0]
            exp_x1;

        reg [COORD_BITS-1:0]
            exp_y1;

        reg [COORD_BITS-1:0]
            exp_x2;

        reg [COORD_BITS-1:0]
            exp_y2;


        reg [CLASS_BITS-1:0]
            exp_cls;


        reg [SCORE_Q_BITS-1:0]
            exp_score;


        reg [ANCHOR_BITS-1:0]
            exp_anchor;


        if (!rst_n) begin

            output_idx =
                0;


            detections_checked =
                0;

            detections_matched =
                0;

            detections_mismatched =
                0;


            field_mismatches =
                0;

            mismatch_printed =
                0;


            first_det_cycle =
                -1;

            last_det_cycle =
                -1;

        end


        else if (det_valid) begin

            // ----------------------------------------------------
            // Timing
            // ----------------------------------------------------

            if (
                output_idx == 0
            ) begin

                first_det_cycle =
                    cycle_count;

            end


            last_det_cycle =
                cycle_count;


            // ----------------------------------------------------
            // Unexpected extra detection
            // ----------------------------------------------------

            if (
                output_idx >=
                expected_n_det
            ) begin

                detections_checked =
                    detections_checked + 1;


                detections_mismatched =
                    detections_mismatched + 1;


                $display("");

                $display(
                    "[FAIL] Unexpected extra detection %0d",
                    output_idx
                );


                $display(
                    "       cls=%0d score=%0d anchor=%0d",
                    det_cls,
                    det_score,
                    det_anchor_idx
                );

            end


            else begin

                // ------------------------------------------------
                // exp_nms.hex:
                //
                // [0] = n_det
                // [1] = overflow
                //
                // then 7 values per detection.
                // ------------------------------------------------

                golden_base =
                    2 +
                    output_idx * 7;


                exp_x1 =
                    nms_mem[
                        golden_base + 0
                    ];


                exp_y1 =
                    nms_mem[
                        golden_base + 1
                    ];


                exp_x2 =
                    nms_mem[
                        golden_base + 2
                    ];


                exp_y2 =
                    nms_mem[
                        golden_base + 3
                    ];


                exp_cls =
                    nms_mem[
                        golden_base + 4
                    ][CLASS_BITS-1:0];


                exp_score =
                    nms_mem[
                        golden_base + 5
                    ][SCORE_Q_BITS-1:0];


                exp_anchor =
                    nms_mem[
                        golden_base + 6
                    ][ANCHOR_BITS-1:0];


                // ------------------------------------------------
                // Exact comparison
                // ------------------------------------------------

                local_errors =
                    0;


                if (
                    det_x1 !==
                    exp_x1
                )
                    local_errors =
                        local_errors + 1;


                if (
                    det_y1 !==
                    exp_y1
                )
                    local_errors =
                        local_errors + 1;


                if (
                    det_x2 !==
                    exp_x2
                )
                    local_errors =
                        local_errors + 1;


                if (
                    det_y2 !==
                    exp_y2
                )
                    local_errors =
                        local_errors + 1;


                if (
                    det_cls !==
                    exp_cls
                )
                    local_errors =
                        local_errors + 1;


                if (
                    det_score !==
                    exp_score
                )
                    local_errors =
                        local_errors + 1;


                if (
                    det_anchor_idx !==
                    exp_anchor
                )
                    local_errors =
                        local_errors + 1;


                // ------------------------------------------------
                // Statistics
                // ------------------------------------------------

                detections_checked =
                    detections_checked + 1;


                field_mismatches =
                    field_mismatches +
                    local_errors;


                if (
                    local_errors == 0
                ) begin

                    detections_matched =
                        detections_matched + 1;


                    $display("");

                    $display(
                        "[PASS] Final detection %0d exact match",
                        output_idx
                    );


                    $display(
                        "       cls=%0d score=%0d anchor=%0d",
                        det_cls,
                        det_score,
                        det_anchor_idx
                    );


                    $display(
                        "       box=(%0d,%0d)-(%0d,%0d)",
                        det_x1,
                        det_y1,
                        det_x2,
                        det_y2
                    );

                end


                else begin

                    detections_mismatched =
                        detections_mismatched + 1;


                    if (
                        mismatch_printed < 20
                    ) begin

                        $display("");

                        $display(
                            "[MISMATCH] Final detection %0d",
                            output_idx
                        );


                        $display(
                            " x1     RTL=%0d GOLDEN=%0d",
                            det_x1,
                            exp_x1
                        );


                        $display(
                            " y1     RTL=%0d GOLDEN=%0d",
                            det_y1,
                            exp_y1
                        );


                        $display(
                            " x2     RTL=%0d GOLDEN=%0d",
                            det_x2,
                            exp_x2
                        );


                        $display(
                            " y2     RTL=%0d GOLDEN=%0d",
                            det_y2,
                            exp_y2
                        );


                        $display(
                            " cls    RTL=%0d GOLDEN=%0d",
                            det_cls,
                            exp_cls
                        );


                        $display(
                            " score  RTL=%0d GOLDEN=%0d",
                            det_score,
                            exp_score
                        );


                        $display(
                            " anchor RTL=%0d GOLDEN=%0d",
                            det_anchor_idx,
                            exp_anchor
                        );


                        $display(
                            " field errors = %0d",
                            local_errors
                        );


                        mismatch_printed =
                            mismatch_printed + 1;

                    end

                end

            end


            output_idx =
                output_idx + 1;

        end

    end


    // ============================================================
    // Main test
    // ============================================================

    initial begin

        // --------------------------------------------------------
        // Initial state
        // --------------------------------------------------------

        rst_n =
            1'b0;


        frame_start =
            1'b0;


        in_valid =
            1'b0;


        anchor_idx =
            '0;


        timeout_counter =
            0;


        first_input_cycle =
            -1;


        last_input_cycle =
            -1;


        done_cycle =
            -1;


        // --------------------------------------------------------
        // Clear input arrays
        // --------------------------------------------------------

        for (
            i = 0;
            i < BBOX_PER_ANCHOR;
            i = i + 1
        ) begin

            bbox_q[i] =
                '0;

        end


        for (
            i = 0;
            i < NUM_CLASSES;
            i = i + 1
        ) begin

            score_q[i] =
                '0;

        end


        // --------------------------------------------------------
        // Reset
        // --------------------------------------------------------

        repeat (5)
            @(posedge clk);


        rst_n =
            1'b1;


        repeat (3)
            @(posedge clk);


        // ========================================================
        // Header
        // ========================================================

        $display("");

        $display(
            "========================================================"
        );

        $display(
            " YOLOV8 MARITIME960 FRAME 0000 THREE-CHECKPOINT TEST"
        );

        $display(
            "========================================================"
        );


        $display(
            " Image                : %0d x %0d",
            IMG_W,
            IMG_H
        );


        $display(
            " Classes              : %0d",
            NUM_CLASSES
        );


        $display(
            " DFL bins             : %0d",
            DFL_BINS
        );


        $display(
            " Raw anchors          : %0d",
            N_ANCHORS
        );


        $display(
            " BBox input values    : %0d",
            TOTAL_BBOX
        );


        $display(
            " Score input values   : %0d",
            TOTAL_SCORE
        );


        $display(
            " Q bits               : %0d",
            Q_BITS
        );


        $display(
            " Score bits           : %0d",
            SCORE_Q_BITS
        );


        $display(
            " Coord bits           : %0d",
            COORD_BITS
        );


        $display(
            " Anchor bits          : %0d",
            ANCHOR_BITS
        );


        $display(
            " Area bits            : %0d",
            AREA_BITS
        );


        $display(
            " Score threshold      : %0d",
            SCORE_Q_THR
        );


        $display(
            " IoU threshold        : %0d/%0d",
            IOU_NUM,
            IOU_DEN
        );


        $display(
            " MAX_CAND             : %0d",
            MAX_CAND
        );


        $display(
            " MAX_DET              : %0d",
            MAX_DET
        );


        $display(
            " Golden detections    : %0d",
            expected_n_det
        );


        $display(
            " Golden overflow      : %0d",
            expected_overflow
        );


        $display(
            " Input mode           : VALID / READY HANDSHAKE"
        );


        $display(
            " Decode architecture  : SHARED MULTI-CYCLE"
        );


        $display(
            " Clock                : 100 MHz"
        );


        $display(
            "--------------------------------------------------------"
        );

        $display("");


        // ========================================================
        // Start frame
        // ========================================================

        @(negedge clk);


        frame_start =
            1'b1;


        @(negedge clk);


        frame_start =
            1'b0;


        // ========================================================
        // Send every raw anchor using valid / ready handshake.
        //
        // The current anchor data is held stable while in_ready=0.
        // The loop advances only after the DUT actually accepts the
        // anchor on a rising clock edge.
        // ========================================================

        for (
            input_anchor = 0;
            input_anchor < N_ANCHORS;
            input_anchor = input_anchor + 1
        ) begin

            // ----------------------------------------------------
            // Present one complete raw anchor.
            // ----------------------------------------------------

            @(negedge clk);


            anchor_idx =
                input_anchor[
                    ANCHOR_BITS-1:0
                ];


            load_input_anchor(
                input_anchor
            );


            in_valid =
                1'b1;


            // ----------------------------------------------------
            // Hold anchor_idx / bbox_q / score_q unchanged until
            // the receiver advertises ready.
            //
            // Sampling ready on falling edges avoids ambiguity with
            // DUT non-blocking state updates on rising edges.
            // ----------------------------------------------------

            while (!in_ready) begin

                @(negedge clk);

            end


            // ----------------------------------------------------
            // Handshake occurs on the next rising edge.
            // ----------------------------------------------------

            @(posedge clk);
            #1;


            if (
                input_anchor == 0
            ) begin

                first_input_cycle =
                    cycle_count;

            end


            if (
                input_anchor ==
                N_ANCHORS - 1
            ) begin

                last_input_cycle =
                    cycle_count;

            end


            if (
                (
                    (input_anchor + 1) %
                    500
                ) == 0
                ||
                (
                    input_anchor ==
                    N_ANCHORS - 1
                )
            ) begin

                $display(
                    "[TB] Raw input progress: %0d / %0d anchors (accepted cycle %0d)",
                    input_anchor + 1,
                    N_ANCHORS,
                    cycle_count
                );

            end


            // ----------------------------------------------------
            // Remove valid only after the accepted rising edge.
            // ----------------------------------------------------

            @(negedge clk);


            in_valid =
                1'b0;

        end


        // ========================================================
        // End raw stream
        // ========================================================

        @(negedge clk);


        in_valid =
            1'b0;


        $display("");

        $display(
            "[TB] All %0d raw anchors accepted through handshake.",
            N_ANCHORS
        );


        $display(
            "[TB] Final Decode transaction draining, then NMS will run automatically..."
        );


        // ========================================================
        // Wait for complete post-processing
        // ========================================================

        timeout_counter =
            0;


        while (
            (done !== 1'b1)
            &&
            (
                timeout_counter <
                2000000
            )
        ) begin

            @(posedge clk);


            timeout_counter =
                timeout_counter + 1;

        end


        if (
            timeout_counter >=
            2000000
        ) begin

            $display("");

            $display(
                "[FAIL] PARAMETERIZED WRAPPER E2E TIMEOUT"
            );


            $display(
                "       n_det currently = %0d",
                n_det
            );


            $stop;

        end


        done_cycle =
            cycle_count;


        // Give output monitor one extra cycle.

        @(posedge clk);


        // ========================================================
        // Final report
        // ========================================================

        $display("");

        $display(
            "========================================================"
        );

        $display(
            " MARITIME960 THREE-CHECKPOINT GOLDEN VERIFICATION"
        );

        $display(
            "========================================================"
        );


        $display(
            " Raw anchors          : %0d",
            N_ANCHORS
        );


        $display(
            " DFL checked          : %0d / %0d",
            dfl_checked,
            N_ANCHORS
        );

        $display(
            " DFL mismatched       : %0d",
            dfl_mismatched
        );

        $display(
            " Decode checked       : %0d / %0d",
            decode_checked,
            N_ANCHORS
        );

        $display(
            " Decode mismatched    : %0d",
            decode_mismatched
        );

        $display(
            "--------------------------------------------------------"
        );

        $display(
            " Golden n_det         : %0d",
            expected_n_det
        );


        $display(
            " RTL n_det            : %0d",
            n_det
        );


        $display(
            " Detections checked   : %0d",
            detections_checked
        );


        $display(
            " Exact matched        : %0d",
            detections_matched
        );


        $display(
            " Mismatched           : %0d",
            detections_mismatched
        );


        $display(
            " Field mismatches     : %0d",
            field_mismatches
        );


        $display(
            " Golden overflow      : %0d",
            expected_overflow
        );


        $display(
            " RTL overflow         : %0d",
            overflow
        );


        $display(
            "--------------------------------------------------------"
        );


        $display(
            " First input cycle    : %0d",
            first_input_cycle
        );


        $display(
            " Last input cycle     : %0d",
            last_input_cycle
        );


        $display(
            " First detection cycle: %0d",
            first_det_cycle
        );


        $display(
            " Last detection cycle : %0d",
            last_det_cycle
        );


        $display(
            " Done cycle           : %0d",
            done_cycle
        );


        $display(
            " Total E2E clocks     : %0d",
            done_cycle -
            first_input_cycle
        );


        $display(
            " Average accepted II : %0f clocks / anchor",
            $itor(last_input_cycle - first_input_cycle) / $itor(N_ANCHORS - 1)
        );


        $display(
            "--------------------------------------------------------"
        );


        // ========================================================
        // PASS / FAIL
        // ========================================================

        if (
            (dfl_checked == N_ANCHORS)
            &&
            (dfl_mismatched == 0)
            &&
            (decode_checked == N_ANCHORS)
            &&
            (decode_mismatched == 0)
            &&
            (
                n_det ==
                expected_n_det
            )
            &&
            (
                detections_checked ==
                expected_n_det
            )
            &&
            (
                detections_mismatched ==
                0
            )
            &&
            (
                field_mismatches ==
                0
            )
            &&
            (
                overflow ==
                expected_overflow
            )
        ) begin

            $display(
                " RESULT               : PASS"
            );


            $display(
                " DFL + DECODE + NMS ALL EXACTLY MATCH GOLDEN"
            );

        end


        else begin

            $display(
                " RESULT               : FAIL"
            );


            $display(
                " PARAMETERIZED WRAPPER DOES NOT MATCH GOLDEN MODEL"
            );

        end


        $display(
            "========================================================"
        );

        $display("");


        repeat (5)
            @(posedge clk);


        $stop;

    end


    // ============================================================
    // Simulation-only TB configuration checks
    // ============================================================

    initial begin

        if (
            BBOX_PER_ANCHOR !=
            4 * DFL_BINS
        ) begin

            $error(
                "TB: BBOX_PER_ANCHOR configuration error"
            );

        end


        if (
            (1 << ANCHOR_BITS) <
            N_ANCHORS
        ) begin

            $error(
                "TB: ANCHOR_BITS too small for N_ANCHORS"
            );

        end


        if (
            (1 << CLASS_BITS) <
            NUM_CLASSES
        ) begin

            $error(
                "TB: CLASS_BITS too small for NUM_CLASSES"
            );

        end

    end

endmodule