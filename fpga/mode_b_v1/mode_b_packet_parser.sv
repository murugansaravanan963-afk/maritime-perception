module mode_b_packet_parser #(
    parameter integer Q_BITS=8, CLASS_BITS=3, ANCHOR_BITS=15, RECORD_BYTES=68
)(
    input logic clk,rst_n,
    input logic byte_valid, output logic byte_ready, input logic [7:0] byte_data,
    output logic frame_start,
    output logic [15:0] frame_id, n_candidates,
    output logic cand_valid, input logic cand_ready,
    output logic [ANCHOR_BITS-1:0] cand_anchor_idx,
    output logic [CLASS_BITS-1:0] cand_cls,
    output logic [7:0] cand_score,
    output logic [Q_BITS-1:0] cand_bbox[0:63],
    output logic packet_done, crc_error, format_error,
    output logic first_byte_pulse, last_payload_pulse
);
    function automatic [15:0] crc_next(input [15:0] crc_in,input [7:0] d);
      integer i; reg [15:0] c; begin c=crc_in^{d,8'h00}; for(i=0;i<8;i=i+1)c=c[15]?((c<<1)^16'h1021):(c<<1); crc_next=c; end
    endfunction
    typedef enum logic[2:0]{S_HDR,S_PAY,S_CRC0,S_CRC1,S_WAIT} st_t;
    st_t st;
    logic [4:0] hidx;
    logic [31:0] payload_len, payload_seen;
    logic [15:0] crc,rx_crc;
    logic [6:0] ridx;
    logic [15:0] rec_count;
    logic pending;
    logic [7:0] hdr[0:15];
    integer i;
    // ------------------------------------------------------------
    // Multi-packet operation
    //
    //  * After packet_done the parser returns to S_HDR (it used to
    //    stay in S_WAIT forever, so only one packet per reset worked).
    //  * In S_HDR the first 4 bytes must be the magic "Y8PP".
    //    Any other byte is dropped, so the 0x00 bytes the host clocks
    //    out on MOSI while reading a result are ignored and the parser
    //    re-synchronises on the next "Y8PP".
    // ------------------------------------------------------------
    localparam logic [7:0] MAGIC0 = 8'h59, MAGIC1 = 8'h38, MAGIC2 = 8'h50, MAGIC3 = 8'h50;
    logic [7:0] magic_exp;
    always_comb begin
      case (hidx[1:0])
        2'd0: magic_exp = MAGIC0;
        2'd1: magic_exp = MAGIC1;
        2'd2: magic_exp = MAGIC2;
        default: magic_exp = MAGIC3;
      endcase
    end
    assign byte_ready = !pending && (st!=S_WAIT);
    assign cand_valid = pending;
    always_ff @(posedge clk or negedge rst_n) begin
      if(!rst_n) begin st<=S_HDR;hidx<=0;payload_len<=0;payload_seen<=0;crc<=16'hffff;rx_crc<=0;ridx<=0;rec_count<=0;pending<=0;frame_start<=0;packet_done<=0;crc_error<=0;format_error<=0;first_byte_pulse<=0;last_payload_pulse<=0;frame_id<=0;n_candidates<=0;cand_anchor_idx<=0;cand_cls<=0;cand_score<=0; end
      else begin
        frame_start<=0; packet_done<=0; first_byte_pulse<=0; last_payload_pulse<=0;
        if(pending && cand_ready) pending<=0;
        if(byte_valid && byte_ready) begin
          case(st)
            S_HDR: begin
              if(hidx<4 && byte_data!=magic_exp) begin
                // Not the expected magic byte: drop it. A 0x59 can start
                // a new magic sequence (handles "YY8PP").
                if(byte_data==MAGIC0) begin
                  crc<=crc_next(16'hffff,byte_data); hdr[0]<=byte_data; hidx<=5'd1;
                  first_byte_pulse<=1; crc_error<=0; format_error<=0;
                end
                else hidx<=5'd0;
              end
              else begin
              if(hidx==0) begin crc<=crc_next(16'hffff,byte_data); first_byte_pulse<=1; crc_error<=0;format_error<=0; end
              else crc<=crc_next(crc,byte_data);
              hdr[hidx]<=byte_data;
              if(hidx==15) begin
                frame_id<={hdr[6],hdr[7]};
                payload_len<={hdr[12],hdr[13],hdr[14],byte_data}; payload_seen<=0;ridx<=0;rec_count<=0;
                if(hdr[0]!=8'h59||hdr[1]!=8'h38||hdr[2]!=8'h50||hdr[3]!=8'h50||hdr[4]!=8'h01||hdr[5]!=8'h03||hdr[10]!=8'h06||hdr[11]!=8'h10||{hdr[12],hdr[13],hdr[14],byte_data}!={hdr[8],hdr[9]}*RECORD_BYTES) begin format_error<=1; st<=S_HDR; packet_done<=1; end
                else begin
                  // n_candidates only changes on a valid header.
                  n_candidates<={hdr[8],hdr[9]};
                  frame_start<=1;
                  // For an empty payload, define the end-of-payload boundary at
                  // header acceptance. This gives the cycle counter a valid
                  // last-byte reference even though there are no payload bytes.
                  if({hdr[8],hdr[9]}==0) last_payload_pulse<=1;
                  st<=({hdr[8],hdr[9]}==0?S_CRC0:S_PAY);
                end
                hidx<=0;
              end else hidx<=hidx+1'b1;
              end
            end
            S_PAY: begin
              crc<=crc_next(crc,byte_data); payload_seen<=payload_seen+1'b1;
              if(ridx==0) cand_anchor_idx[14:8]<=byte_data[6:0];
              else if(ridx==1) cand_anchor_idx[7:0]<=byte_data;
              else if(ridx==2) cand_cls<=byte_data[CLASS_BITS-1:0];
              else if(ridx==3) cand_score<=byte_data;
              else cand_bbox[ridx-4]<=byte_data;
              if(payload_seen+1==payload_len) last_payload_pulse<=1;
              if(ridx==RECORD_BYTES-1) begin pending<=1; ridx<=0; rec_count<=rec_count+1'b1; if(payload_seen+1==payload_len) st<=S_CRC0; end
              else ridx<=ridx+1'b1;
            end
            S_CRC0: begin rx_crc[15:8]<=byte_data; st<=S_CRC1; end
            S_CRC1: begin rx_crc[7:0]<=byte_data; if({rx_crc[15:8],byte_data}!=crc) crc_error<=1; packet_done<=1; st<=S_HDR; hidx<=0; end
            default: st<=S_HDR;
          endcase
        end
      end
    end
endmodule
