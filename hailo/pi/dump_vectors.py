#!/usr/bin/env python3
"""Dump the raw quantized Hailo outputs of maritime6_960.hef for the RTL (DFL + NMS) team.

  python3 dump_vectors.py --hef maritime6_960.hef --images test/images --frames 30 --out fpga_vectors

Writes exactly the format rtl_handoff/golden_model.py --vectors reads:
  meta.json         input size, classes, output names/shapes, qp_scale, qp_zp, bits, dequant check
  frame_XXXX.npz    bbox_q (18900, 64) and score_q (18900, 6), native integers from the Hailo
  frame_XXXX.png    the letterboxed 960x960 input (BGR) for viewing results
Frames are picked so that every class appears, then filled in file order.
"""
import argparse,json
from pathlib import Path
import numpy as np
import cv2
import maritime_post as mp

def pick(images,labels,n):
    files=sorted(p for p in images.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'));chosen=[];seen=set()
    for p in files:   # first pass: cover every class
        cls={int(l.split()[0]) for l in open(labels/(p.stem+'.txt')).read().split('\n') if l.strip()} if (labels/(p.stem+'.txt')).exists() else set()
        if cls-seen and len(chosen)<n:chosen.append(p);seen|=cls
    for p in files:
        if len(chosen)>=n:break
        if p not in chosen:chosen.append(p)
    return chosen

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hef',required=True);ap.add_argument('--images',type=Path,required=True)
    ap.add_argument('--labels',type=Path,default=None);ap.add_argument('--frames',type=int,default=30);ap.add_argument('--out',type=Path,default=Path('fpga_vectors'))
    a=ap.parse_args()
    from hailo_platform import HEF,VDevice,FormatType,HailoStreamInterface,ConfigureParams,InputVStreamParams,OutputVStreamParams,InferVStreams
    a.out.mkdir(parents=True,exist_ok=True)
    hef=HEF(a.hef);ins=hef.get_input_vstream_infos();outs=hef.get_output_vstream_infos()
    sizes={o.name:int(np.prod(o.shape)) for o in outs}
    bo=next(o for o in outs if sizes[o.name]==mp.N_ANCHORS*64);so=next(o for o in outs if sizes[o.name]==mp.N_ANCHORS*6)
    labels=a.labels or a.images.parent/'labels'
    files=pick(a.images,labels,a.frames) if labels.exists() else sorted(a.images.iterdir())[:a.frames]
    with VDevice(VDevice.create_params()) as dev:
        ng=dev.configure(hef,ConfigureParams.create_from_hef(hef,interface=HailoStreamInterface.PCIe))[0]
        ip=InputVStreamParams.make(ng,format_type=FormatType.UINT8)
        native=OutputVStreamParams.make(ng,format_type=FormatType.AUTO);flt=OutputVStreamParams.make(ng,format_type=FormatType.FLOAT32)
        frames=[]
        with ng.activate():
            with InferVStreams(ng,ip,native) as pipe:
                for k,p in enumerate(files):
                    rgb,_,_=mp.letterbox(cv2.imread(str(p)));x=np.ascontiguousarray(rgb[None])
                    r=pipe.infer({ins[0].name:x});bq=np.asarray(r[bo.name]).reshape(mp.N_ANCHORS,64);sq=np.asarray(r[so.name]).reshape(mp.N_ANCHORS,6)
                    if np.issubdtype(bq.dtype,np.floating):raise SystemExit('AUTO returned float; use FormatType.UINT8 for the outputs')
                    np.savez(a.out/f'frame_{k:04d}.npz',bbox_q=bq,score_q=sq);cv2.imwrite(str(a.out/f'frame_{k:04d}.png'),cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR))
                    frames.append({'name':f'frame_{k:04d}','source':p.name});print('frame',k,bq.dtype,bq.shape,sq.shape,flush=True)
                    if k==0:first=(x,bq.copy(),sq.copy())
            with InferVStreams(ng,ip,flt) as pipe:r=pipe.infer({ins[0].name:first[0]})
    diffs={}
    for o,q in ((bo,first[1]),(so,first[2])):
        deq=np.float32(o.quant_info.qp_scale)*(q.astype(np.float32)-np.float32(o.quant_info.qp_zp))
        diffs[o.name]=float(np.abs(deq.reshape(-1)-np.asarray(r[o.name],np.float32).reshape(-1)).max())
    q=lambda o,arr:{'name':o.name,'shape':[int(s) for s in o.shape],'format':str(o.format.type),'qp_scale':float(o.quant_info.qp_scale),
                    'qp_zp':float(o.quant_info.qp_zp),'bits':8*arr.dtype.itemsize,'dtype':str(arr.dtype)}
    meta={'hef':str(Path(a.hef).resolve()),'input_w':mp.SIZE,'input_h':mp.SIZE,'n_anchors':mp.N_ANCHORS,'num_classes':6,'class_names':mp.NAMES,
          'bbox':q(bo,first[1]),'score':q(so,first[2]),'dequant_check_max_abs_diff':diffs,'frames':frames,
          'preprocess':'Ultralytics letterbox to 960x960 (pad 114), BGR->RGB, uint8 0-255; /255 is inside the HEF',
          'anchor_order':'stride 8 (120x120), stride 16 (60x60), stride 32 (30x30); row-major; centre ((x+0.5)*s, (y+0.5)*s)'}
    (a.out/'meta.json').write_text(json.dumps(meta,indent=2)+'\n')
    print('dequant check (should be ~0):',diffs);print('Saved',len(frames),'frames to',a.out)
if __name__=='__main__':main()
