#!/usr/bin/env python3
"""Fake virtual FPGA for testing pi_mode_b_client.py without ModelSim.

Speaks the same TCP framing and Y8PP Mode B packets as the RTL team's mode_b_bridge.py, but answers
with golden_model.fixed_ref run on the received candidates only (like the RTL's sparse decode).
Cycle counters in the reply are 0: this is not a timing model.

  python3 mock_virtual_fpga.py                       # listens on 127.0.0.1:9000
  python3 mock_virtual_fpga.py --host 0.0.0.0        # reachable from the Pi
"""
import argparse,socket
import numpy as np
import maritime_post as mp
import golden_model as gm
import mode_b_protocol as pb

# maritime6_960.hef output quantization (same values as the RTL's rtl_params.vh / dfl_exp_lut.mem)
QUANT=dict(bbox_scale=0.17380952835083008,bbox_zp=171.0,score_scale=0.003921568859368563,score_zp=0.0,qbits=8,sbits=8)

def answer(packet,model,params):
    req,err=pb.parse_packet(packet)
    if err:
        status=0x02 if err.startswith('crc') else 0x04
        return pb.build_reply(0,[],status=status),err
    bq=np.zeros((model.N,64),np.uint8);sq=np.zeros((model.N,pb.NUM_CLASSES),np.uint8)
    bq[req['anchor']]=req['bbox'];sq[req['anchor'],req['cls']]=req['score']
    dets,n_cand,overflow=pb.golden_dets(gm,model,params,bq,sq)
    status=0x01|(0x08 if overflow else 0)|(0x10 if len(dets)==pb.MAX_DET else 0)
    return pb.build_reply(req['frame_id'],dets,status=status),f"frame {req['frame_id']}: {len(req['anchor'])} candidates -> {len(dets)} detections"

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=9000)
    ap.add_argument('--once',action='store_true',help='exit after one request (for tests)');a=ap.parse_args()
    model=gm.Model(mp.SIZE,mp.SIZE,pb.NUM_CLASSES,pb.MAX_CAND,pb.MAX_DET);params=gm.make_params(QUANT,0.25,0.45)
    with socket.socket(socket.AF_INET,socket.SOCK_STREAM) as srv:
        srv.setsockopt(socket.SOL_SOCKET,socket.SO_REUSEADDR,1);srv.bind((a.host,a.port));srv.listen(1)
        print(f'[mock fpga] listening on {a.host}:{a.port}',flush=True)
        while True:
            conn,addr=srv.accept()
            with conn:
                try:
                    reply,msg=answer(pb.recv_msg(conn),model,params);pb.send_msg(conn,reply)
                    print(f'[mock fpga] {addr[0]}: {msg}',flush=True)
                except Exception as e:
                    print(f'[mock fpga] {addr[0]}: ERROR {e}',flush=True)
                    try:conn.sendall(b'\0\0\0\0')
                    except OSError:pass
            if a.once:break

if __name__=='__main__':main()
