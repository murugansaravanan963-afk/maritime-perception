#!/usr/bin/env python3
"""Laptop reference: run the CUT ONNX (same two raw outputs as the HEF) through maritime_post on the
test images. Its mAP is the float baseline the Pi HEF result is compared against.

  python onnx_reference.py --onnx maritime6_960_raw.onnx --images test/images --labels test/labels --out onnx_reference.json
"""
import argparse,json,time
from pathlib import Path
import numpy as np
import cv2
import maritime_post as mp

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--onnx',required=True);ap.add_argument('--images',type=Path,required=True)
    ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--conf',type=float,default=0.001);ap.add_argument('--iou',type=float,default=0.7);a=ap.parse_args()
    import onnxruntime as ort
    s=ort.InferenceSession(a.onnx,providers=['CPUExecutionProvider']);gts={};preds={}
    files=sorted(p for p in a.images.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    t0=time.time()
    for k,p in enumerate(files):
        img=cv2.imread(str(p));h,w=img.shape[:2];rgb,r,pad=mp.letterbox(img)
        x=(rgb.astype(np.float32)/255.0).transpose(2,0,1)[None]   # the HEF does this /255 on chip
        bbox,score=s.run(None,{'images':x})
        b,sc,c=mp.decode(mp.channel_major_to_anchor_major(bbox,64),mp.channel_major_to_anchor_major(score,6),a.conf,a.iou)
        b=mp.unletterbox(b,r,pad,w,h)
        preds[p.stem]=[(int(ci),float(si),*map(float,bi)) for bi,si,ci in zip(b,sc,c)]
        gts[p.stem]=mp.read_labels(a.labels/(p.stem+'.txt'),w,h)
        if k%100==0:print(k,len(files),f'{time.time()-t0:.0f}s',flush=True)
    res={'engine':'onnxruntime float32 (cut ONNX, laptop)','images':len(files),'conf':a.conf,'iou':a.iou,**mp.evaluate(gts,preds)}
    a.out.write_text(json.dumps(res,indent=2)+'\n')
    print(f"mAP50 {res['mAP50']:.4f}  mAP50-95 {res['mAP50_95']:.4f}");print({k:round(v['AP50'],3) for k,v in res['per_class'].items() if v})
if __name__=='__main__':main()
