`timescale 1ns/1ps
module yolov8_mode_b_frame0000_tb;
 `include "rtl_params.vh"
 localparam integer NB=N_ANCHORS*64, NS=N_ANCHORS*NUM_CLASSES;
 logic clk=0,rst_n=0,rx_valid=0,rx_ready;logic[7:0]rx_data=0;
 logic result_ready,rs=0,tx_ready=1,tx_valid;logic[7:0]tx_data;logic tx_last;
 logic[7:0] bbox_mem[0:NB-1],score_mem[0:NS-1];
 integer cand_idx[0:MAX_CAND-1]; integer cand_cls[0:MAX_CAND-1]; integer cand_score[0:MAX_CAND-1]; integer nc=0;
 integer gx1[0:MAX_DET-1],gy1[0:MAX_DET-1],gx2[0:MAX_DET-1],gy2[0:MAX_DET-1],gcls[0:MAX_DET-1],gsc[0:MAX_DET-1],gai[0:MAX_DET-1]; integer gnd,gov;
 logic[7:0] reply[0:1213]; integer rn=0;
 always #3.333 clk=~clk;
 yolov8_mode_b_packet_top dut(.clk(clk),.rst_n(rst_n),.rx_valid(rx_valid),.rx_ready(rx_ready),.rx_data(rx_data),.result_ready(result_ready),.result_read_start(rs),.tx_ready(tx_ready),.tx_valid(tx_valid),.tx_data(tx_data),.tx_last(tx_last));
 function automatic[15:0] crc_next(input[15:0]ci,input[7:0]d);integer i;reg[15:0]c;begin c=ci^{d,8'h00};for(i=0;i<8;i=i+1)c=c[15]?((c<<1)^16'h1021):(c<<1);crc_next=c;end endfunction
 task automatic sendb(input[7:0]b);begin @(negedge clk);rx_data=b;rx_valid=1;do @(posedge clk);while(!rx_ready);@(negedge clk);rx_valid=0;end endtask
 task automatic send_crc_byte(input[7:0]b,inout[15:0]c);begin sendb(b);c=crc_next(c,b);end endtask
 integer i,j,c,best,fd,r; logic[15:0] crc; integer plen; reg[7:0] hb[0:15];
 initial begin
   $readmemh("frame_0000/in_bbox.hex",bbox_mem); $readmemh("frame_0000/in_score.hex",score_mem);
   for(i=0;i<N_ANCHORS;i=i+1) begin best=score_mem[i*NUM_CLASSES]; c=0; for(j=1;j<NUM_CLASSES;j=j+1) if(score_mem[i*NUM_CLASSES+j]>best) begin best=score_mem[i*NUM_CLASSES+j]; c=j; end if(best>=SCORE_Q_THR) begin cand_idx[nc]=i; cand_cls[nc]=c; cand_score[nc]=best; nc=nc+1; end end
   fd=$fopen("frame_0000/exp_nms.hex","r"); r=$fscanf(fd,"%h %d",gnd,gov);for(i=0;i<gnd;i=i+1)r=$fscanf(fd,"%h %h %h %h %h %h %h",gx1[i],gy1[i],gx2[i],gy2[i],gcls[i],gsc[i],gai[i]);$fclose(fd);
   $display("[TB] Mode-B frame0000 candidates=%0d golden_det=%0d",nc,gnd); if(nc==0) $fatal(1,"Mode-B prefilter produced zero candidates; input/argmax check failed"); else begin $display("[TB] First candidate: anchor=%0d cls=%0d score=%0d",cand_idx[0],cand_cls[0],cand_score[0]); $display("[TB] Last  candidate: anchor=%0d cls=%0d score=%0d",cand_idx[nc-1],cand_cls[nc-1],cand_score[nc-1]); end
   repeat(8)@(posedge clk);rst_n=1;repeat(3)@(posedge clk);
   plen=68*nc; hb[0]=8'h59;hb[1]=8'h38;hb[2]=8'h50;hb[3]=8'h50;hb[4]=1;hb[5]=8'h03;hb[6]=0;hb[7]=0;hb[8]=nc>>8;hb[9]=nc;hb[10]=NUM_CLASSES;hb[11]=16;hb[12]=plen>>24;hb[13]=plen>>16;hb[14]=plen>>8;hb[15]=plen;
   crc=16'hffff;for(i=0;i<16;i=i+1)send_crc_byte(hb[i],crc);
   for(i=0;i<nc;i=i+1) begin send_crc_byte(cand_idx[i]>>8,crc);send_crc_byte(cand_idx[i],crc);send_crc_byte(cand_cls[i],crc);send_crc_byte(cand_score[i],crc);for(j=0;j<64;j=j+1)send_crc_byte(bbox_mem[cand_idx[i]*64+j],crc);end
   sendb(crc[15:8]);sendb(crc[7:0]);
   wait(result_ready); $display("[TB] Compute done; reading reply..."); @(negedge clk);rs=1;@(negedge clk);rs=0;
   rn=0; while(!tx_last) begin @(posedge clk);if(tx_valid)begin reply[rn]=tx_data;rn=rn+1;end end @(posedge clk);if(tx_valid)begin reply[rn]=tx_data;rn=rn+1;end
   $display("[TB] reply bytes=%0d status=%02x n_det=%0d",rn,reply[0],reply[3]);
   if(reply[0][0]!==1'b1 || reply[3]!==gnd) $fatal(1,"Mode-B status/n_det mismatch");
   for(i=0;i<gnd;i=i+1) begin integer p,ax1,ay1,ax2,ay2,acl,asc,aai;p=12+12*i;ax1={reply[p],reply[p+1]};ay1={reply[p+2],reply[p+3]};ax2={reply[p+4],reply[p+5]};ay2={reply[p+6],reply[p+7]};acl=reply[p+8];asc=reply[p+9];aai={reply[p+10],reply[p+11]}; if(ax1!=gx1[i]||ay1!=gy1[i]||ax2!=gx2[i]||ay2!=gy2[i]||acl!=gcls[i]||asc!=gsc[i]||aai!=gai[i]) $fatal(1,"Detection %0d mismatch",i); end
   $display("========================================================");$display(" MODE-B 68B FRAME0000 PACKET -> V6 -> REPLY : PASS");$display(" candidates=%0d detections=%0d",nc,gnd);$display("========================================================");$stop;
 end
endmodule
