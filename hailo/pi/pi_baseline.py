#!/usr/bin/env python3
"""Baseline of maritime6_960.hef on Raspberry Pi 5 + Hailo-8L: accuracy and speed.

  python3 pi_baseline.py --hef maritime6_960.hef --images test/images --labels test/labels

1. Accuracy: every test image -> HEF (float32 outputs) -> maritime_post.decode (conf 0.001,
   NMS IoU 0.7, the same code as onnx_reference.py) -> mAP50 / mAP50-95. Compare with the
   laptop ONNX reference to see the Hailo INT8 quantization loss.
2. Speed (deployment settings conf 0.25, NMS IoU 0.45): per-stage time for each image
   (letterbox, Hailo inference, CPU post-processing) and frames per second.
Writes pi_baseline_results.json and a few overlay images to --out.
"""
import argparse,json,platform,subprocess,time
from pathlib import Path
import numpy as np
import cv2
import maritime_post as mp

def cmd(args):
    try:return subprocess.run(args,capture_output=True,text=True,timeout=20).stdout.strip()
    except Exception as e:return f'unavailable: {e}'

def state():
    return {'temp':cmd(['vcgencmd','measure_temp']),'arm_clock':cmd(['vcgencmd','measure_clock','arm']),'throttled':cmd(['vcgencmd','get_throttled'])}

def stats(x):
    x=np.asarray(x,float);return {'median_ms':float(np.median(x)),'p95_ms':float(np.percentile(x,95)),'mean_ms':float(x.mean())}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--hef',required=True);ap.add_argument('--images',type=Path,required=True)
    ap.add_argument('--labels',type=Path,required=True);ap.add_argument('--out',type=Path,default=Path('pi_baseline'))
    ap.add_argument('--overlays',type=int,default=12);a=ap.parse_args()
    from hailo_platform import HEF,VDevice,FormatType,HailoStreamInterface,ConfigureParams,InputVStreamParams,OutputVStreamParams,InferVStreams
    a.out.mkdir(parents=True,exist_ok=True)
    hef=HEF(a.hef);ins=hef.get_input_vstream_infos();outs=hef.get_output_vstream_infos()
    desc=lambda i:{'name':i.name,'shape':[int(s) for s in i.shape],'format':str(i.format.type),'qp_scale':float(i.quant_info.qp_scale),'qp_zp':float(i.quant_info.qp_zp)}
    meta={'hef':str(Path(a.hef).resolve()),'inputs':[desc(i) for i in ins],'outputs':[desc(o) for o in outs]}
    print(json.dumps(meta,indent=1),flush=True)
    sizes={o.name:int(np.prod(o.shape)) for o in outs}
    bbox_name=next(n for n,s in sizes.items() if s==mp.N_ANCHORS*64);score_name=next(n for n,s in sizes.items() if s==mp.N_ANCHORS*6)
    files=sorted(p for p in a.images.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))
    gts={};preds={};t_pre=[];t_inf=[];t_post=[];t_all=[];over=0
    before=state()
    with VDevice(VDevice.create_params()) as dev:
        ng=dev.configure(hef,ConfigureParams.create_from_hef(hef,interface=HailoStreamInterface.PCIe))[0]
        # On-chip normalization: the HEF takes 0-255 uint8 RGB. Do NOT divide by 255 here.
        ip=InputVStreamParams.make(ng,format_type=FormatType.UINT8);op=OutputVStreamParams.make(ng,format_type=FormatType.FLOAT32)
        with ng.activate(),InferVStreams(ng,ip,op) as pipe:
            for _ in range(10):pipe.infer({ins[0].name:np.zeros((1,960,960,3),np.uint8)})   # warm-up
            for k,p in enumerate(files):
                img=cv2.imread(str(p));h,w=img.shape[:2]
                t0=time.perf_counter();rgb,r,pad=mp.letterbox(img);x=np.ascontiguousarray(rgb[None])
                t1=time.perf_counter();res=pipe.infer({ins[0].name:x})
                t2=time.perf_counter()
                bbox=mp.to_anchor_major(res[bbox_name],64);score=mp.to_anchor_major(res[score_name],6)
                b,sc,c=mp.decode(bbox,score,0.25,0.45)                     # deployment settings (timed)
                t3=time.perf_counter()
                t_pre.append((t1-t0)*1e3);t_inf.append((t2-t1)*1e3);t_post.append((t3-t2)*1e3);t_all.append((t3-t0)*1e3)
                be,se,ce=mp.decode(bbox,score,0.001,0.7)                    # evaluation settings (not timed)
                be=mp.unletterbox(be,r,pad,w,h)
                preds[p.stem]=[(int(ci),float(si),*map(float,bi)) for bi,si,ci in zip(be,se,ce)]
                gts[p.stem]=mp.read_labels(a.labels/(p.stem+'.txt'),w,h)
                if over<a.overlays and len(b):
                    view=img.copy()
                    for bi,si,ci in zip(mp.unletterbox(b,r,pad,w,h),sc,c):
                        x1,y1,x2,y2=map(int,bi);cv2.rectangle(view,(x1,y1),(x2,y2),(0,255,0),2)
                        cv2.putText(view,f'{mp.NAMES[ci]} {si:.2f}',(x1,max(15,y1-4)),cv2.FONT_HERSHEY_SIMPLEX,.6,(0,255,0),2)
                    cv2.imwrite(str(a.out/f'overlay_{over:02d}.jpg'),view);over+=1
                if k%50==0:print(f'{k}/{len(files)}  infer {t_inf[-1]:.1f} ms',flush=True)
    acc=mp.evaluate(gts,preds)
    res={'meta':meta,'images':len(files),'platform':platform.platform(),'hailortcli_identify':cmd(['hailortcli','fw-control','identify']),
         'state_before':before,'state_after':state(),
         'accuracy':{'conf':0.001,'nms_iou':0.7,**acc},
         'timing_deploy_settings':{'conf':0.25,'nms_iou':0.45,'preprocess_letterbox':stats(t_pre),'hailo_inference':stats(t_inf),
                                   'cpu_postprocess_dfl_decode_nms':stats(t_post),'end_to_end':stats(t_all),
                                   'fps_end_to_end_single_stream':1000/float(np.median(t_all))},
         'note':'Inference is blocking, one frame at a time. Hardware-only throughput: run `hailortcli run maritime6_960.hef`.'}
    (a.out/'pi_baseline_results.json').write_text(json.dumps(res,indent=2)+'\n')
    T=res['timing_deploy_settings']
    print(f"\nACCURACY  mAP50 {acc['mAP50']:.4f}  mAP50-95 {acc['mAP50_95']:.4f}")
    for n,v in acc['per_class'].items():print(f'  {n:18s} AP50 {v["AP50"]:.3f}  AP50-95 {v["AP50_95"]:.3f}' if v else f'  {n}: no labels')
    print(f"TIMING (median ms)  letterbox {T['preprocess_letterbox']['median_ms']:.1f}  hailo {T['hailo_inference']['median_ms']:.1f}  "
          f"post {T['cpu_postprocess_dfl_decode_nms']['median_ms']:.1f}  total {T['end_to_end']['median_ms']:.1f}  -> {T['fps_end_to_end_single_stream']:.1f} FPS")
    print('Saved',a.out/'pi_baseline_results.json')
if __name__=='__main__':main()
