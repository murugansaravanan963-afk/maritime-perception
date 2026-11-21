#!/usr/bin/env python3
"""mAP50 under a minimum-object-size (detection range) rule, COCO-style ignore.

Ground-truth boxes whose short side, scaled to the training size (longest image side =
--scale px), is below the threshold are "don't care": a prediction matching one is
neither a true nor a false positive, and missing one is not penalised. Unmatched
predictions below the threshold are also dropped. Training labels are unchanged; the
same rule must be applied to val and test. Threshold 0 = the ordinary score.

  python scripts/eval_min_size.py --release releases/maritime6_pilot_r1 \
      --weights runs/detect/r1_y8s_1280/weights/best.pt --imgsz 1280 --split val \
      --thresholds 0 4 6 8 10 12 16 --out reports/training/r1_y8s_1280_min_size_val.json
"""
import argparse,json
from pathlib import Path
import numpy as np
from release_gate import NAMES,digest,read_jsonl

def iou(a,b):
 """IoU of one box a (4,) against boxes b (n,4), xyxy."""
 if not len(b):return np.zeros(0)
 iw=np.clip(np.minimum(a[2],b[:,2])-np.maximum(a[0],b[:,0]),0,None);ih=np.clip(np.minimum(a[3],b[:,3])-np.maximum(a[1],b[:,1]),0,None)
 inter=iw*ih;return inter/((a[2]-a[0])*(a[3]-a[1])+(b[:,2]-b[:,0])*(b[:,3]-b[:,1])-inter)

def ap101(tp,conf,n_gt):
 """COCO 101-point interpolated AP from per-prediction TP flags."""
 if n_gt==0:return None
 if not len(tp):return 0.0
 o=np.argsort(-np.asarray(conf),kind='stable');tp=np.asarray(tp,float)[o]
 ctp=np.cumsum(tp);cfp=np.cumsum(1-tp);rec=ctp/n_gt;prec=ctp/np.maximum(ctp+cfp,1e-12)
 prec=np.maximum.accumulate(prec[::-1])[::-1]
 return float(np.mean([prec[np.searchsorted(rec,r,side='left')] if r<=rec[-1] else 0.0 for r in np.linspace(0,1,101)]))

def score(gts,preds,thr,scale,iou_thr=0.5):
 """gts/preds: {image_id: (W,H,[(cls,x1,y1,x2,y2)...] or [(cls,conf,x1,..)...])}."""
 out={}
 for c in range(len(NAMES)):
  tps=[];confs=[];n_gt=0
  for k,(W,H,g) in gts.items():
   s=scale/max(W,H)
   gb=np.array([b[1:] for b in g if b[0]==c],float).reshape(-1,4)
   care=(np.minimum(gb[:,2]-gb[:,0],gb[:,3]-gb[:,1])*s>=thr) if len(gb) else np.zeros(0,bool)
   n_gt+=int(care.sum());used=np.zeros(len(gb),bool)
   for p in sorted((p for p in preds.get(k,[]) if p[0]==c),key=lambda p:-p[1]):
    box=np.array(p[2:],float);ov=iou(box,gb)
    # Prefer an unmatched cared-for box, then an ignored one (COCO order).
    best=-1
    for want in (True,False):
     cand=np.where((ov>=iou_thr)&~used&(care==want))[0] if len(gb) else []
     if len(cand):best=int(cand[np.argmax(ov[cand])]);break
    if best>=0:
     used[best]=True
     if care[best]:tps.append(1);confs.append(p[1])
     continue
    if min(box[2]-box[0],box[3]-box[1])*s<thr:continue
    tps.append(0);confs.append(p[1])
  out[NAMES[c]]={'AP50':ap101(tps,confs,n_gt),'gt_counted':n_gt}
 vals=[v['AP50'] for v in out.values() if v['AP50'] is not None]
 return {'mAP50':float(np.mean(vals)) if vals else None,'per_class':out}

def load_gts(release,split):
 gts={}
 for r in read_jsonl(release/'manifest.jsonl'):
  if r['split']!=split:continue
  im=r['image'];stem=digest(r['id'].encode())[:32];suffix=(im.get('export') or {}).get('suffix') or Path(im['path']).suffix.lower()
  gts[str(release/'images'/split/(stem+suffix))]=(im['width'],im['height'],[(b['class_id'],*b['xyxy']) for b in r['boxes']])
 return gts

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--release',type=Path,required=True);ap.add_argument('--weights',type=Path,required=True)
 ap.add_argument('--imgsz',type=int,default=1280);ap.add_argument('--split',default='val');ap.add_argument('--thresholds',type=float,nargs='+',default=[0,4,6,8,10,12,16])
 ap.add_argument('--device',default='0');ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.split=='test':raise SystemExit('Choose the rule on val. Run test once, at the very end, with the rule already fixed.')
 release=a.release.resolve();gts=load_gts(release,a.split)
 from ultralytics import YOLO
 model=YOLO(str(a.weights));preds={}
 paths=sorted(gts)
 for i in range(0,len(paths),16):
  # Same settings as Ultralytics val: conf 0.001, NMS IoU 0.7, up to 300 boxes.
  for path,res in zip(paths[i:i+16],model.predict(paths[i:i+16],imgsz=a.imgsz,conf=0.001,iou=0.7,max_det=300,device=a.device,verbose=False)):
   preds[path]=[(int(c),float(s),*map(float,b)) for b,c,s in zip(res.boxes.xyxy.tolist(),res.boxes.cls.tolist(),res.boxes.conf.tolist())]
 result={'release':str(release),'weights':str(a.weights),'weights_sha256':digest(a.weights.read_bytes()),'split':a.split,'imgsz':a.imgsz,
         'rule':'ignore objects whose short side, scaled so the longest image side = imgsz, is below the threshold (px)','images':len(gts),
         'by_threshold':{str(t):score(gts,preds,t,a.imgsz) for t in a.thresholds}}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n')
 print(f"{'min px':>7} {'mAP50':>6} "+' '.join(f'{n[:8]:>8}' for n in NAMES))
 for t,v in result['by_threshold'].items():
  print(f"{t:>7} {v['mAP50']:6.3f} "+' '.join(f"{v['per_class'][n]['AP50'] if v['per_class'][n]['AP50'] is not None else float('nan'):8.3f}" for n in NAMES))
 print('Saved',a.out)
if __name__=='__main__':main()
