`timescale 1ns / 1ps

// ============================================================================
// Maritime960 V6 candidate-scaling stress benchmark
//
// Purpose:
//   Measure DFL+Decode, NMS, and total cycles for candidate counts
//   10 / 100 / 1000 / 4096 without modifying the verified V6 RTL.
//
// Test definition:
//   - N_ANCHORS is set equal to the requested candidate count for each DUT.
//   - Every supplied anchor passes SCORE_Q_THR.
//   - All candidates use class 0 and equal high score.
//   - DFL logits strongly favor bin 0 on every side, producing very small boxes.
//     This avoids mass suppression and makes NMS repeatedly scan the candidate
//     table until MAX_DET is reached. This is a synthetic stress benchmark,
//     not a real-frame latency distribution.
//
// Timing boundaries:
//   first_accept  : first (in_valid && in_ready)
//   decode_end    : final candidate completes Decode (dut.decode_frame_end)
//   done          : NMS done
//
// Reported:
//   DFL+Decode cycles = decode_end - first_accept
//   NMS cycles        = done - decode_end
//   Total cycles      = done - first_accept
//   Latency           = cycles / 150 MHz (post-route verified V6 clock)
// ============================================================================

module maritime_candidate_case #(
    parameter integer TEST_N = 10,
    parameter integer CASE_ID = 0
)(
    input  logic clk,
    input  logic start,
    output logic finished,
    output longint dfl_decode_cycles,
    output longint nms_cycles,
    output longint total_cycles,
    output integer final_n_det,
    output logic final_overflow
);

    `include "rtl_params.vh"

    localparam integer DFL_SIDES = 4;
    localparam integer LUT_WIDTH = 16;
    localparam integer OFF_BITS  = 12;

    logic rst_n;
    logic frame_start;
    logic in_valid;
    logic in_ready;
    logic [ANCHOR_BITS-1:0] anchor_idx;
    logic [Q_BITS-1:0] bbox_q [0:DFL_SIDES*DFL_BINS-1];
    logic [SCORE_Q_BITS-1:0] score_q [0:NUM_CLASSES-1];

    logic det_valid;
    logic [COORD_BITS-1:0] det_x1, det_y1, det_x2, det_y2;
    logic [CLASS_BITS-1:0] det_cls;
    logic [SCORE_Q_BITS-1:0] det_score;
    logic [ANCHOR_BITS-1:0] det_anchor_idx;
    logic [$clog2(MAX_DET+1)-1:0] n_det;
    logic overflow;
    logic done;

    longint cycle_count;
    longint first_accept_cycle;
    longint decode_end_cycle;
    longint done_cycle;
    integer accepted;
    integer k;
    integer c;

    yolov8_postprocess_top #(
        .Q_BITS(Q_BITS),
        .DFL_BINS(DFL_BINS),
        .DFL_SIDES(DFL_SIDES),
        .LUT_WIDTH(LUT_WIDTH),
        .LUT_DEPTH(LUT_DEPTH),
        .LUT_FILE("dfl_exp_lut.mem"),
        .OFF_BITS(OFF_BITS),
        .OFF_FRAC(OFF_FRAC),
        .NUM_CLASSES(NUM_CLASSES),
        .SCORE_BITS(SCORE_Q_BITS),
        .CLASS_BITS(CLASS_BITS),
        .SCORE_Q_THR(SCORE_Q_THR),
        .IMG_W(IMG_W),
        .IMG_H(IMG_H),
        .COORD_BITS(COORD_BITS),
        .COORD_FRAC(COORD_FRAC),
        .ANCHOR_BITS(ANCHOR_BITS),
        .N_ANCHORS(TEST_N),
        .MAX_CAND(MAX_CAND),
        .MAX_DET(MAX_DET),
        .IOU_NUM(IOU_NUM),
        .IOU_DEN(IOU_DEN),
        .AREA_BITS(AREA_BITS)
    ) dut (
        .clk(clk),
        .rst_n(rst_n),
        .frame_start(frame_start),
        .in_valid(in_valid),
        .in_ready(in_ready),
        .anchor_idx(anchor_idx),
        .bbox_q(bbox_q),
        .score_q(score_q),
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

    always @(posedge clk) begin
        cycle_count <= cycle_count + 1;
    end

    initial begin
        rst_n = 1'b0;
        frame_start = 1'b0;
        in_valid = 1'b0;
        anchor_idx = '0;
        finished = 1'b0;
        dfl_decode_cycles = 0;
        nms_cycles = 0;
        total_cycles = 0;
        final_n_det = 0;
        final_overflow = 1'b0;
        cycle_count = 0;
        first_accept_cycle = -1;
        decode_end_cycle = -1;
        done_cycle = -1;
        accepted = 0;

        for (k = 0; k < DFL_SIDES*DFL_BINS; k = k + 1)
            bbox_q[k] = '0;
        for (c = 0; c < NUM_CLASSES; c = c + 1)
            score_q[c] = '0;

        $display("[CASE %0d] N=%0d waiting for start...", CASE_ID, TEST_N);
        wait(start === 1'b1);
        $display("[CASE %0d] N=%0d start received at t=%0t", CASE_ID, TEST_N, $time);

        // Fresh reset for this independent case.
        repeat (8) @(posedge clk);
        rst_n = 1'b1;
        $display("[CASE %0d] reset released", CASE_ID);
        repeat (3) @(posedge clk);

        // One-cycle frame_start before first anchor.
        frame_start = 1'b1;
        @(posedge clk);
        frame_start = 1'b0;

        accepted = 0;
        $display("[TEST] Starting N=%0d candidates", TEST_N);
        while (accepted < TEST_N) begin
            // Strong bin-0 DFL distribution on all four sides.
            // qmax-q for bins 1..15 clamps to the tail of the LUT.
            for (k = 0; k < DFL_SIDES*DFL_BINS; k = k + 1)
                bbox_q[k] = '0;
            bbox_q[0]  = {Q_BITS{1'b1}};
            bbox_q[16] = {Q_BITS{1'b1}};
            bbox_q[32] = {Q_BITS{1'b1}};
            bbox_q[48] = {Q_BITS{1'b1}};

            // Every anchor passes threshold; same class and same score creates
            // a deterministic, scan-heavy NMS stress case.
            for (c = 0; c < NUM_CLASSES; c = c + 1)
                score_q[c] = '0;
            score_q[0] = {SCORE_Q_BITS{1'b1}};

            anchor_idx = accepted[ANCHOR_BITS-1:0];
            in_valid = 1'b1;

            @(posedge clk);
            while (!in_ready) @(posedge clk);

            if (first_accept_cycle < 0)
                first_accept_cycle = cycle_count;

            accepted = accepted + 1;
        end

        in_valid = 1'b0;
        $display("[CASE %0d] N=%0d all candidates accepted; waiting for final Decode", CASE_ID, TEST_N);

        // Final Decode completion marks NMS collection end.
        wait(dut.decode_frame_end === 1'b1);
        decode_end_cycle = cycle_count;
        $display("[CASE %0d] N=%0d Decode complete; waiting for NMS done", CASE_ID, TEST_N);

        wait(done === 1'b1);
        done_cycle = cycle_count;

        dfl_decode_cycles = decode_end_cycle - first_accept_cycle;
        nms_cycles = done_cycle - decode_end_cycle;
        total_cycles = done_cycle - first_accept_cycle;
        final_n_det = n_det;
        final_overflow = overflow;

        $display("[CASE %0d] N=%0d DFL+Decode=%0d NMS=%0d Total=%0d n_det=%0d overflow=%0d",
                 CASE_ID, TEST_N, dfl_decode_cycles, nms_cycles,
                 total_cycles, final_n_det, final_overflow);

        finished = 1'b1;
    end

    // Per-case watchdog: prevents silent hangs during bring-up.
    initial begin
        wait(start === 1'b1);
        repeat (20000000) @(posedge clk);
        if (!finished) begin
            $display("[ERROR] CASE %0d N=%0d timeout: accepted=%0d in_ready=%b done=%b",
                     CASE_ID, TEST_N, accepted, in_ready, done);
            $fatal(1);
        end
    end

endmodule


module yolov8_maritime960_candidate_scaling_tb;

    localparam real CLK_NS = 10.0;      // simulation clock only
    localparam real REPORT_MHZ = 150.0; // verified XC7A100T post-route clock

    logic clk;
    logic start10, start100, start1000, start4096;
    logic fin10, fin100, fin1000, fin4096;

    longint dd10, dd100, dd1000, dd4096;
    longint nm10, nm100, nm1000, nm4096;
    longint tt10, tt100, tt1000, tt4096;
    integer nd10, nd100, nd1000, nd4096;
    logic ov10, ov100, ov1000, ov4096;

    real dd_ms10, dd_ms100, dd_ms1000, dd_ms4096;
    real nm_ms10, nm_ms100, nm_ms1000, nm_ms4096;
    real tt_ms10, tt_ms100, tt_ms1000, tt_ms4096;

    initial clk = 1'b0;
    always #(CLK_NS/2.0) clk = ~clk;

    maritime_candidate_case #(.TEST_N(10), .CASE_ID(0)) u10 (
        .clk(clk), .start(start10), .finished(fin10),
        .dfl_decode_cycles(dd10), .nms_cycles(nm10), .total_cycles(tt10),
        .final_n_det(nd10), .final_overflow(ov10));

    maritime_candidate_case #(.TEST_N(100), .CASE_ID(1)) u100 (
        .clk(clk), .start(start100), .finished(fin100),
        .dfl_decode_cycles(dd100), .nms_cycles(nm100), .total_cycles(tt100),
        .final_n_det(nd100), .final_overflow(ov100));

    maritime_candidate_case #(.TEST_N(1000), .CASE_ID(2)) u1000 (
        .clk(clk), .start(start1000), .finished(fin1000),
        .dfl_decode_cycles(dd1000), .nms_cycles(nm1000), .total_cycles(tt1000),
        .final_n_det(nd1000), .final_overflow(ov1000));

    maritime_candidate_case #(.TEST_N(4096), .CASE_ID(3)) u4096 (
        .clk(clk), .start(start4096), .finished(fin4096),
        .dfl_decode_cycles(dd4096), .nms_cycles(nm4096), .total_cycles(tt4096),
        .final_n_det(nd4096), .final_overflow(ov4096));

    task automatic pulse_start(ref logic s);
        begin
            s = 1'b1;
            @(posedge clk);
            s = 1'b0;
        end
    endtask

    initial begin
        start10 = 0; start100 = 0; start1000 = 0; start4096 = 0;

        $display("");
        $display("========================================================");
        $display(" MARITIME960 V6 CANDIDATE-SCALING STRESS BENCHMARK");
        $display("========================================================");
        $display(" Candidate counts : 10 / 100 / 1000 / 4096");
        $display(" Report clock     : 150 MHz (XC7A100T timing closed)");
        $display(" Workload         : all candidates pass, class 0, equal score");
        $display(" Purpose          : DFL+Decode + NMS scaling stress");
        $display("--------------------------------------------------------");

        repeat (5) @(posedge clk);
        pulse_start(start10);
        wait(fin10);

        repeat (10) @(posedge clk);
        pulse_start(start100);
        wait(fin100);

        repeat (10) @(posedge clk);
        pulse_start(start1000);
        wait(fin1000);

        repeat (10) @(posedge clk);
        pulse_start(start4096);
        wait(fin4096);

        dd_ms10   = dd10   / (REPORT_MHZ * 1000.0);
        dd_ms100  = dd100  / (REPORT_MHZ * 1000.0);
        dd_ms1000 = dd1000 / (REPORT_MHZ * 1000.0);
        dd_ms4096 = dd4096 / (REPORT_MHZ * 1000.0);

        nm_ms10   = nm10   / (REPORT_MHZ * 1000.0);
        nm_ms100  = nm100  / (REPORT_MHZ * 1000.0);
        nm_ms1000 = nm1000 / (REPORT_MHZ * 1000.0);
        nm_ms4096 = nm4096 / (REPORT_MHZ * 1000.0);

        tt_ms10   = tt10   / (REPORT_MHZ * 1000.0);
        tt_ms100  = tt100  / (REPORT_MHZ * 1000.0);
        tt_ms1000 = tt1000 / (REPORT_MHZ * 1000.0);
        tt_ms4096 = tt4096 / (REPORT_MHZ * 1000.0);

        $display("");
        $display("========================================================");
        $display(" CANDIDATE-SCALING RESULTS @150 MHz");
        $display("========================================================");
        $display(" Cand   DFL+Dec cyc   NMS cyc      Total cyc    Total ms   n_det  ovf");
        $display("-----------------------------------------------------------------------");
        $display("   10   %10d   %10d   %10d   %8.6f   %5d   %0d", dd10, nm10, tt10, tt_ms10, nd10, ov10);
        $display("  100   %10d   %10d   %10d   %8.6f   %5d   %0d", dd100, nm100, tt100, tt_ms100, nd100, ov100);
        $display(" 1000   %10d   %10d   %10d   %8.6f   %5d   %0d", dd1000, nm1000, tt1000, tt_ms1000, nd1000, ov1000);
        $display(" 4096   %10d   %10d   %10d   %8.6f   %5d   %0d", dd4096, nm4096, tt4096, tt_ms4096, nd4096, ov4096);
        $display("-----------------------------------------------------------------------");
        $display(" DFL+Decode ms: 10=%0.6f 100=%0.6f 1000=%0.6f 4096=%0.6f", dd_ms10, dd_ms100, dd_ms1000, dd_ms4096);
        $display(" NMS ms       : 10=%0.6f 100=%0.6f 1000=%0.6f 4096=%0.6f", nm_ms10, nm_ms100, nm_ms1000, nm_ms4096);
        $display("========================================================");
        $display(" NOTE: synthetic scan-heavy stress case; not real-frame distribution.");
        $display("========================================================");
        $display("");

        $stop;
    end

    // Generous watchdog for the 4096-candidate NMS stress case.
    initial begin
        #500000000; // 500 ms simulation time at 1 ns units
        $fatal(1, "Candidate-scaling benchmark watchdog timeout");
    end

endmodule
