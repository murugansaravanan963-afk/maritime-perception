`timescale 1ns / 1ps

module yolov8_postprocess_real_30frame_benchmark_tb;

    `include "rtl_params.vh"

    // ============================================================
    // Test configuration
    // ============================================================

    localparam integer DFL_SIDES = 4;
    localparam integer LUT_WIDTH = 16;
    localparam integer OFF_BITS  = 12;

    localparam integer NUM_FRAMES = 30;

    localparam integer BBOX_VALUES =
        N_ANCHORS * DFL_SIDES * DFL_BINS;

    localparam integer SCORE_VALUES =
        N_ANCHORS * NUM_CLASSES;

    localparam integer GOLDEN_WORDS =
        2 + MAX_DET * 7;

    localparam integer DFL_GOLDEN_FIELDS = 4;
    localparam integer DECODE_GOLDEN_FIELDS = 7;
    localparam integer TOTAL_DFL_GOLDEN = N_ANCHORS * DFL_GOLDEN_FIELDS;
    localparam integer TOTAL_DECODE_GOLDEN = N_ANCHORS * DECODE_GOLDEN_FIELDS;

    // ============================================================
    // Clock / reset
    // ============================================================

    logic clk;
    logic rst_n;

    logic frame_start;

    // ============================================================
    // DUT input
    // ============================================================

    logic in_valid;
    logic in_ready;

    logic [ANCHOR_BITS-1:0]
        anchor_idx;

    logic [Q_BITS-1:0]
        bbox_q [0:DFL_SIDES*DFL_BINS-1];

    logic [SCORE_Q_BITS-1:0]
        score_q [0:NUM_CLASSES-1];

    // ============================================================
    // DUT output
    // ============================================================

    logic det_valid;

    logic [COORD_BITS-1:0] det_x1;
    logic [COORD_BITS-1:0] det_y1;
    logic [COORD_BITS-1:0] det_x2;
    logic [COORD_BITS-1:0] det_y2;

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
        bbox_mem [0:BBOX_VALUES-1];

    logic [SCORE_Q_BITS-1:0]
        score_mem [0:SCORE_VALUES-1];

    logic [OFF_BITS-1:0]
        dfl_golden_mem [0:TOTAL_DFL_GOLDEN-1];

    logic [COORD_BITS-1:0]
        decode_golden_mem [0:TOTAL_DECODE_GOLDEN-1];

    // ============================================================
    // Golden NMS memory
    //
    // File format:
    //
    // word 0 = n_det
    // word 1 = overflow
    //
    // then 7 words per detection:
    //
    // x1
    // y1
    // x2
    // y2
    // cls
    // score
    // anchor
    // ============================================================

    logic [31:0]
        golden_mem [0:GOLDEN_WORDS-1];

    // ============================================================
    // Captured RTL detections
    // ============================================================

    logic [COORD_BITS-1:0]
        rtl_x1 [0:MAX_DET-1];

    logic [COORD_BITS-1:0]
        rtl_y1 [0:MAX_DET-1];

    logic [COORD_BITS-1:0]
        rtl_x2 [0:MAX_DET-1];

    logic [COORD_BITS-1:0]
        rtl_y2 [0:MAX_DET-1];

    logic [CLASS_BITS-1:0]
        rtl_cls [0:MAX_DET-1];

    logic [SCORE_Q_BITS-1:0]
        rtl_score [0:MAX_DET-1];

    logic [ANCHOR_BITS-1:0]
        rtl_anchor [0:MAX_DET-1];

    // ============================================================
    // Hardware benchmark clock / CPU reference
    //
    // Cycle counts are measured by RTL simulation. FPGA_CLK_MHZ is
    // used only to convert those cycle counts into hardware latency.
    // 100 MHz is used only as a provisional conversion for V6 simulation.
    // Replace with the measured V6 implementation clock after Vivado timing closure.
    // ============================================================

    localparam real FPGA_CLK_MHZ = 100.0;

    localparam real CPU_CPP_MEDIAN_MS = 2.910;
    localparam real CPU_CPP_P95_MS    = 2.910; // placeholder: replace with Maritime p95 if provided

    // ============================================================
    // Per-frame timing results
    // ============================================================

    integer frame_cycles [0:NUM_FRAMES-1];
    integer sorted_cycles [0:NUM_FRAMES-1];

    integer frame_pass [0:NUM_FRAMES-1];

    integer frame_dfl_mismatch [0:NUM_FRAMES-1];
    integer frame_decode_mismatch [0:NUM_FRAMES-1];
    integer frame_dfl_checked [0:NUM_FRAMES-1];
    integer frame_decode_checked [0:NUM_FRAMES-1];

    // ============================================================
    // Counters / statistics
    // ============================================================

    integer cycle_count;

    integer det_capture_count;

    integer frame_index;
    integer anchor_index;

    integer i;
    integer j;

    integer failed_frames;
    integer passed_frames;

    integer total_detections;

    integer first_input_cycle;
    integer last_input_cycle;
    integer done_cycle;

    integer current_frame_cycles;

    integer golden_n_det;
    integer golden_overflow;

    integer mismatches;

    integer dfl_checked;
    integer dfl_mismatched;
    integer dfl_mismatch_printed;
    integer decode_checked;
    integer decode_mismatched;
    integer decode_mismatch_printed;

    integer total_dfl_mismatches;
    integer total_decode_mismatches;

    integer tmp;

    real median_cycles;
    real p95_cycles;

    real median_ms;
    real p95_ms;

    real cpp_median_speedup;
    real cpp_p95_speedup;

    // ============================================================
    // File names
    //
    // Large enough for:
    //
    // frame_0000/in_bbox.hex
    // ============================================================

    string bbox_file;
    string score_file;
    string nms_file;
    string dfl_file;
    string decode_file;

    // ============================================================
    // Clock
    // ============================================================

    initial begin
        clk = 1'b0;

        forever
            #5 clk = ~clk;
    end

    // ============================================================
    // Global cycle counter
    // ============================================================

    always @(posedge clk) begin

        if (!rst_n)
            cycle_count <= 0;
        else
            cycle_count <= cycle_count + 1;

    end

    // ============================================================
    // DUT
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
    // Detection capture
    // ============================================================

    always @(posedge clk) begin

        if (!rst_n) begin

            det_capture_count <= 0;

        end
        else begin

            if (det_valid) begin

                if (
                    det_capture_count <
                    MAX_DET
                ) begin

                    rtl_x1[
                        det_capture_count
                    ] <= det_x1;

                    rtl_y1[
                        det_capture_count
                    ] <= det_y1;

                    rtl_x2[
                        det_capture_count
                    ] <= det_x2;

                    rtl_y2[
                        det_capture_count
                    ] <= det_y2;

                    rtl_cls[
                        det_capture_count
                    ] <= det_cls;

                    rtl_score[
                        det_capture_count
                    ] <= det_score;

                    rtl_anchor[
                        det_capture_count
                    ] <= det_anchor_idx;

                end

                det_capture_count <=
                    det_capture_count + 1;

            end

        end

    end

    // ============================================================
    // Build zero-padded frame file names
    //
    // ModelSim version being used does not reliably implement
    // %04d zero-padding in $sformatf.
    //
    // Frames are only 0..29, so construct them explicitly.
    // ============================================================

    task automatic build_frame_filenames(
        input integer idx
    );

        begin

            if (idx < 10) begin

                bbox_file =
                    $sformatf(
                        "frame_000%0d/in_bbox.hex",
                        idx
                    );

                score_file =
                    $sformatf(
                        "frame_000%0d/in_score.hex",
                        idx
                    );

                nms_file = $sformatf("frame_000%0d/exp_nms.hex", idx);
                dfl_file = $sformatf("frame_000%0d/exp_dfl.hex", idx);
                decode_file = $sformatf("frame_000%0d/exp_decode.hex", idx);

            end
            else begin

                bbox_file =
                    $sformatf(
                        "frame_00%0d/in_bbox.hex",
                        idx
                    );

                score_file =
                    $sformatf(
                        "frame_00%0d/in_score.hex",
                        idx
                    );

                nms_file = $sformatf("frame_00%0d/exp_nms.hex", idx);
                dfl_file = $sformatf("frame_00%0d/exp_dfl.hex", idx);
                decode_file = $sformatf("frame_00%0d/exp_decode.hex", idx);

            end

        end

    endtask

    // ============================================================
    // Load one real frame
    // ============================================================

    task automatic load_frame_files(
        input integer idx
    );

        begin

            build_frame_filenames(
                idx
            );

            $display("");
            $display(
                "[TB] Frame %0d loading vectors...",
                idx
            );

            $display(
                "[TB] Loading bbox : %s",
                bbox_file
            );

            $display(
                "[TB] Loading score: %s",
                score_file
            );

            $display("[TB] Loading DFL   : %s", dfl_file);
            $display("[TB] Loading Decode: %s", decode_file);
            $display("[TB] Loading NMS   : %s", nms_file);

            // ----------------------------------------------------
            // Clear memories before loading.
            // ----------------------------------------------------

            for (
                i = 0;
                i < BBOX_VALUES;
                i = i + 1
            ) begin

                bbox_mem[i] = '0;

            end

            for (
                i = 0;
                i < SCORE_VALUES;
                i = i + 1
            ) begin

                score_mem[i] = '0;

            end

            for (i = 0; i < TOTAL_DFL_GOLDEN; i = i + 1) dfl_golden_mem[i] = '0;
            for (i = 0; i < TOTAL_DECODE_GOLDEN; i = i + 1) decode_golden_mem[i] = '0;

            for (
                i = 0;
                i < GOLDEN_WORDS;
                i = i + 1
            ) begin

                golden_mem[i] = '0;

            end

            // ----------------------------------------------------
            // Load vectors.
            // ----------------------------------------------------

            $readmemh(
                bbox_file,
                bbox_mem
            );

            $readmemh(
                score_file,
                score_mem
            );

            $readmemh(dfl_file, dfl_golden_mem);
            $readmemh(decode_file, decode_golden_mem);

            $readmemh(
                nms_file,
                golden_mem
            );

            golden_n_det =
                golden_mem[0];

            golden_overflow =
                golden_mem[1];

            $display(
                "[TB] Frame %0d golden: n_det=%0d overflow=%0d",
                idx,
                golden_n_det,
                golden_overflow
            );

        end

    endtask

    // ============================================================
    // Drive one raw anchor into input buses
    // ============================================================

    task automatic load_anchor_to_ports(
        input integer a
    );

        integer q;

        begin

            anchor_idx =
                a[ANCHOR_BITS-1:0];

            for (
                q = 0;
                q < DFL_SIDES * DFL_BINS;
                q = q + 1
            ) begin

                bbox_q[q] =
                    bbox_mem[
                        a * DFL_SIDES * DFL_BINS
                        + q
                    ];

            end

            for (
                q = 0;
                q < NUM_CLASSES;
                q = q + 1
            ) begin

                score_q[q] =
                    score_mem[
                        a * NUM_CLASSES
                        + q
                    ];

            end

        end

    endtask

    // ============================================================
    // Send all 3549 anchors using VALID/READY handshake
    // ============================================================

    task automatic send_all_anchors;

        begin

            first_input_cycle =
                -1;

            last_input_cycle =
                -1;

            for (
                anchor_index = 0;
                anchor_index < N_ANCHORS;
                anchor_index = anchor_index + 1
            ) begin

                // ------------------------------------------------
                // Prepare current anchor.
                // ------------------------------------------------

                @(negedge clk);

                load_anchor_to_ports(
                    anchor_index
                );

                in_valid =
                    1'b1;

                // ------------------------------------------------
                // Hold data stable until handshake succeeds.
                // ------------------------------------------------

                while (!in_ready) begin

                    @(posedge clk);

                    #1;

                end

                // ------------------------------------------------
                // Actual acceptance occurs at the next rising edge
                // for which valid and ready are both high.
                //
                // Since in_valid was asserted on the preceding
                // falling edge and ready is already high here,
                // this rising edge is the handshake edge.
                // ------------------------------------------------

                @(posedge clk);

                #1;

                if (
                    first_input_cycle < 0
                ) begin

                    first_input_cycle =
                        cycle_count;

                end

                last_input_cycle =
                    cycle_count;

                // ------------------------------------------------
                // Remove valid after successful acceptance.
                // ------------------------------------------------

                @(negedge clk);

                in_valid =
                    1'b0;

                // ------------------------------------------------
                // Progress output
                // ------------------------------------------------

                if (
                    ((anchor_index + 1) % 500) == 0
                    ||
                    anchor_index ==
                    N_ANCHORS - 1
                ) begin

                    $display(
                        "[TB] Frame %0d progress: %0d / %0d anchors",
                        frame_index,
                        anchor_index + 1,
                        N_ANCHORS
                    );

                end

            end

        end

    endtask

    // ============================================================
    // Compare current frame against exp_nms.hex
    // ============================================================

    task automatic compare_current_frame;

        integer d;

        integer base;

        begin

            mismatches =
                0;

            // ----------------------------------------------------
            // Detection count
            // ----------------------------------------------------

            if (
                n_det !== golden_n_det
            ) begin

                $display(
                    "[FAIL] Frame %0d n_det RTL=%0d GOLDEN=%0d",
                    frame_index,
                    n_det,
                    golden_n_det
                );

                mismatches =
                    mismatches + 1;

            end

            // ----------------------------------------------------
            // Overflow
            // ----------------------------------------------------

            if (
                overflow !== golden_overflow[0]
            ) begin

                $display(
                    "[FAIL] Frame %0d overflow RTL=%0d GOLDEN=%0d",
                    frame_index,
                    overflow,
                    golden_overflow
                );

                mismatches =
                    mismatches + 1;

            end

            // ----------------------------------------------------
            // Number of det_valid pulses
            // ----------------------------------------------------

            if (
                det_capture_count !=
                golden_n_det
            ) begin

                $display(
                    "[FAIL] Frame %0d det_valid count RTL=%0d GOLDEN=%0d",
                    frame_index,
                    det_capture_count,
                    golden_n_det
                );

                mismatches =
                    mismatches + 1;

            end

            // ----------------------------------------------------
            // Exact field comparison
            // ----------------------------------------------------

            for (
                d = 0;
                d < golden_n_det;
                d = d + 1
            ) begin

                base =
                    2 + d * 7;

                if (
                    rtl_x1[d] !==
                    golden_mem[
                        base + 0
                    ][COORD_BITS-1:0]
                ) begin

                    mismatches =
                        mismatches + 1;

                end

                if (
                    rtl_y1[d] !==
                    golden_mem[
                        base + 1
                    ][COORD_BITS-1:0]
                ) begin

                    mismatches =
                        mismatches + 1;

                end

                if (
                    rtl_x2[d] !==
                    golden_mem[
                        base + 2
                    ][COORD_BITS-1:0]
                ) begin

                    mismatches =
                        mismatches + 1;

                end

                if (
                    rtl_y2[d] !==
                    golden_mem[
                        base + 3
                    ][COORD_BITS-1:0]
                ) begin

                    mismatches =
                        mismatches + 1;

                end

                if (
                    rtl_cls[d] !==
                    golden_mem[
                        base + 4
                    ][CLASS_BITS-1:0]
                ) begin

                    mismatches =
                        mismatches + 1;

                end

                if (
                    rtl_score[d] !==
                    golden_mem[
                        base + 5
                    ][SCORE_Q_BITS-1:0]
                ) begin

                    mismatches =
                        mismatches + 1;

                end

                if (
                    rtl_anchor[d] !==
                    golden_mem[
                        base + 6
                    ][ANCHOR_BITS-1:0]
                ) begin

                    mismatches =
                        mismatches + 1;

                end

            end

            // ----------------------------------------------------
            // Frame result
            // ----------------------------------------------------

            if (
                (mismatches == 0) &&
                (dfl_mismatched == 0) &&
                (decode_mismatched == 0) &&
                (dfl_checked == N_ANCHORS) &&
                (decode_checked == N_ANCHORS)
            ) begin

                frame_pass[
                    frame_index
                ] =
                    1;

                passed_frames =
                    passed_frames + 1;

                $display(
                    "[PASS] Frame %0d exact match: n_det=%0d cycles=%0d latency=%0f ms",
                    frame_index,
                    n_det,
                    current_frame_cycles,
                    current_frame_cycles / (FPGA_CLK_MHZ * 1000.0)
                );

            end
            else begin

                frame_pass[
                    frame_index
                ] =
                    0;

                failed_frames =
                    failed_frames + 1;

                $display(
                    "[FAIL] Frame %0d NMS_errors=%0d cycles=%0d",
                    frame_index,
                    mismatches,
                    current_frame_cycles
                );
                $display(
                    "       DFL    checked=%0d/%0d mismatched=%0d",
                    dfl_checked, N_ANCHORS, dfl_mismatched
                );
                $display(
                    "       Decode checked=%0d/%0d mismatched=%0d",
                    decode_checked, N_ANCHORS, decode_mismatched
                );
                $display(
                    "       NMS    det_valid=%0d golden_n_det=%0d overflow RTL/GOLDEN=%0d/%0d",
                    det_capture_count, golden_n_det, overflow, golden_overflow
                );

            end

            total_detections =
                total_detections +
                n_det;

        end

    endtask

    // ============================================================
    // Run one complete frame
    // ============================================================

    task automatic run_one_frame(
        input integer idx
    );

        begin

            frame_index =
                idx;

            // ----------------------------------------------------
            // Load real vectors.
            // ----------------------------------------------------

            load_frame_files(
                idx
            );

            // ----------------------------------------------------
            // Reset DUT between frames.
            // ----------------------------------------------------

            rst_n =
                1'b0;

            frame_start =
                1'b0;

            in_valid =
                1'b0;

            anchor_idx =
                '0;

            repeat (5)
                @(posedge clk);

            rst_n =
                1'b1;

            repeat (3)
                @(posedge clk);

            // ----------------------------------------------------
            // Start frame.
            // ----------------------------------------------------

            @(negedge clk);

            frame_start =
                1'b1;

            @(negedge clk);

            frame_start =
                1'b0;

            // ----------------------------------------------------
            // Send all parameterized anchors.
            // ----------------------------------------------------

            send_all_anchors();

            $display(
                "[TB] Frame %0d all anchors accepted. Waiting for NMS...",
                idx
            );

            // ----------------------------------------------------
            // Wait for complete frame.
            // ----------------------------------------------------

            while (!done) begin

                @(posedge clk);

                #1;

            end

            done_cycle =
                cycle_count;

            // ----------------------------------------------------
            // Exact timing definition:
            //
            // first accepted raw anchor
            //        ->
            // final NMS done
            // ----------------------------------------------------

            current_frame_cycles =
                done_cycle -
                first_input_cycle;

            frame_cycles[
                idx
            ] =
                current_frame_cycles;

            frame_dfl_checked[idx] = dfl_checked;
            frame_decode_checked[idx] = decode_checked;
            frame_dfl_mismatch[idx] = dfl_mismatched;
            frame_decode_mismatch[idx] = decode_mismatched;

            // ----------------------------------------------------
            // Allow final captured detection NBA updates to settle.
            // ----------------------------------------------------

            #1;

            compare_current_frame();

            // ----------------------------------------------------
            // Small gap before next reset/frame.
            // ----------------------------------------------------

            repeat (3)
                @(posedge clk);

        end

    endtask

    // ============================================================
    // Main benchmark
    // ============================================================

    initial begin

        rst_n =
            1'b0;

        frame_start =
            1'b0;

        in_valid =
            1'b0;

        anchor_idx =
            '0;

        cycle_count =
            0;

        det_capture_count =
            0;

        passed_frames =
            0;

        failed_frames =
            0;

        total_detections =
            0;

        dfl_checked = 0;
        dfl_mismatched = 0;
        dfl_mismatch_printed = 0;
        decode_checked = 0;
        decode_mismatched = 0;
        decode_mismatch_printed = 0;
        total_dfl_mismatches = 0;
        total_decode_mismatches = 0;

        for (i = 0; i < NUM_FRAMES; i = i + 1) begin
            frame_cycles[i] = 0;
            sorted_cycles[i] = 0;
            frame_pass[i] = 0;
            frame_dfl_mismatch[i] = 0;
            frame_decode_mismatch[i] = 0;
            frame_dfl_checked[i] = 0;
            frame_decode_checked[i] = 0;
        end

        // --------------------------------------------------------
        // Initialize ports.
        // --------------------------------------------------------

        for (
            i = 0;
            i < DFL_SIDES * DFL_BINS;
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

        $display("");
        $display(
            "========================================================"
        );
        $display(
            " YOLOV8 30-FRAME REAL FPGA BENCHMARK"
        );
        $display(
            "========================================================"
        );
        $display(
            " Frames              : %0d",
            NUM_FRAMES
        );
        $display(
            " Anchors/frame       : %0d",
            N_ANCHORS
        );
        $display(
            " Clock               : %0.0f MHz",
            FPGA_CLK_MHZ
        );
        $display(
            " Timing boundary     : first accepted anchor -> done"
        );
        $display(
            " Input mode          : VALID / READY"
        );
        $display(
            "--------------------------------------------------------"
        );

        // ========================================================
        // Run all 30 frames
        // ========================================================

        for (
            frame_index = 0;
            frame_index < NUM_FRAMES;
            frame_index = frame_index + 1
        ) begin

            run_one_frame(
                frame_index
            );

        end

        // ========================================================
        // Copy timing array for sorting
        // ========================================================

        for (
            i = 0;
            i < NUM_FRAMES;
            i = i + 1
        ) begin

            sorted_cycles[i] =
                frame_cycles[i];

        end

        // ========================================================
        // Simple ascending bubble sort
        // ========================================================

        for (
            i = 0;
            i < NUM_FRAMES - 1;
            i = i + 1
        ) begin

            for (
                j = 0;
                j < NUM_FRAMES - 1 - i;
                j = j + 1
            ) begin

                if (
                    sorted_cycles[j] >
                    sorted_cycles[j+1]
                ) begin

                    tmp =
                        sorted_cycles[j];

                    sorted_cycles[j] =
                        sorted_cycles[j+1];

                    sorted_cycles[j+1] =
                        tmp;

                end

            end

        end

        // ========================================================
        // Statistics
        //
        // 30 samples:
        //
        // Median:
        // average of sorted index 14 and 15
        //
        // p95 nearest-rank:
        // ceil(0.95 * 30) = 29th sample
        // zero-based index = 28
        // ========================================================

        median_cycles =
            (
                sorted_cycles[14]
                +
                sorted_cycles[15]
            )
            /
            2.0;

        p95_cycles =
            sorted_cycles[28];

        median_ms =
            median_cycles /
            (FPGA_CLK_MHZ * 1000.0);

        p95_ms =
            p95_cycles /
            (FPGA_CLK_MHZ * 1000.0);

        cpp_median_speedup =
            CPU_CPP_MEDIAN_MS /
            median_ms;

        cpp_p95_speedup =
            CPU_CPP_P95_MS /
            p95_ms;

        // ========================================================
        // Per-frame table
        // ========================================================

        $display("");
        $display(
            "========================================================"
        );
        $display(
            " PER-FRAME MARITIME960 RTL RESULTS"
        );
        $display(
            "========================================================"
        );
        $display(
            " Frame    Cycles       Latency(ms)     Result"
        );
        $display(
            "--------------------------------------------------------"
        );

        for (
            i = 0;
            i < NUM_FRAMES;
            i = i + 1
        ) begin

            if (
                frame_pass[i]
            ) begin

                $display(
                    " %0d       %0d       %0f       PASS",
                    i,
                    frame_cycles[i],
                    frame_cycles[i] / (FPGA_CLK_MHZ * 1000.0)
                );

            end
            else begin

                $display(
                    " %0d       %0d       %0f       FAIL",
                    i,
                    frame_cycles[i],
                    frame_cycles[i] / (FPGA_CLK_MHZ * 1000.0)
                );

            end

        end

        // ========================================================
        // Final benchmark summary
        // ========================================================

        $display("");
        $display(
            "========================================================"
        );
        $display(
            " MARITIME960 30-FRAME THREE-CHECKPOINT BENCHMARK SUMMARY"
        );
        $display(
            "========================================================"
        );

        $display(
            " Frames passed       : %0d / %0d",
            passed_frames,
            NUM_FRAMES
        );

        $display(
            " Frames failed       : %0d",
            failed_frames
        );

        total_dfl_mismatches = 0;
        total_decode_mismatches = 0;
        for (i = 0; i < NUM_FRAMES; i = i + 1) begin
            total_dfl_mismatches = total_dfl_mismatches + frame_dfl_mismatch[i];
            total_decode_mismatches = total_decode_mismatches + frame_decode_mismatch[i];
        end
        $display(" Total DFL mismatches   : %0d", total_dfl_mismatches);
        $display(" Total Decode mismatches: %0d", total_decode_mismatches);

        $display(
            " Total detections    : %0d",
            total_detections
        );

        $display(
            "--------------------------------------------------------"
        );

        $display(
            " Min cycles          : %0d",
            sorted_cycles[0]
        );

        $display(
            " Median cycles       : %0f",
            median_cycles
        );

        $display(
            " P95 cycles          : %0f",
            p95_cycles
        );

        $display(
            " Max cycles          : %0d",
            sorted_cycles[NUM_FRAMES-1]
        );

        $display(
            "--------------------------------------------------------"
        );

        $display(
            " FPGA median @%0.0fMHz : %0f ms",
            FPGA_CLK_MHZ,
            median_ms
        );

        $display(
            " FPGA p95    @%0.0fMHz : %0f ms",
            FPGA_CLK_MHZ,
            p95_ms
        );

        $display(
            "--------------------------------------------------------"
        );

        $display(" V6 latency conversion above is provisional @100MHz.");
        $display(" Recompute ms/speedup after Maritime V6 Vivado timing closure.");
        $display(" CPU Maritime full-frame C++ median reference: 2.910 ms");
        $display("--------------------------------------------------------");

        if (
            failed_frames == 0
        ) begin

            $display(
                " RESULT              : PASS"
            );

            $display(
                " ALL 30 MARITIME960 FRAMES EXACTLY MATCH DFL + DECODE + NMS GOLDEN"
            );

        end
        else begin

            $display(
                " RESULT              : FAIL"
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

endmodule