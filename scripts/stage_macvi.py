#!/usr/bin/env python3
"""Stage MaCVi 2025 USV distance-estimation TRAIN frames as navigation_marker candidates.

Publisher labels are YOLO-normalized nav_buoy boxes with distance as a 6th column.
Vessels in these frames are NOT labelled by the publisher, so every staged frame
still needs full-frame review (assist_review.py) before it can be used.
Frame numbers are continuous inside a recording; a jump > 30 starts a new recording.
Every STRIDE-th frame is kept so adjacent near-identical frames are not credited.
"""
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from stage_candidates import ROOT,make_record,check_image
from audit_candidates import sha,write_json
STRIDE=8
BASE=ROOT/'raw/macvi_usv_dist';EX=BASE/'extracted'

def main():
 out=ROOT/'staging/candidates/macvi_usv_dist.jsonl'
 if out.exists():raise FileExistsError('Refuse to overwrite staged inventory')
 archive_sha=sha((BASE/'macvi25_usv_dist_train.zip').read_bytes())
 frames=sorted(int(p.stem) for p in (EX/'images').glob('*.png'))
 rec=0;prev=None;groups={}
 for n in frames:
  if prev is not None and n-prev>30:rec+=1
  groups[n]=rec;prev=n
 pos={};tasks=[]
 for n in frames:
  g=groups[n];pos[g]=pos.get(g,-1)+1
  if pos[g]%STRIDE:continue
  name=f'{n:06d}';lp=EX/'labels'/(name+'.txt');lb=lp.read_bytes()
  r=make_record('macvi:'+name,'macvi_usv_dist',sha(lb),EX/'images'/(name+'.png'),source_split='train',group=f'proposal:macvi:recording_{g}',view='boat')
  r['source_revision']='macvi2025_usv_dist_train_zip:'+archive_sha
  r.update(native_image_id=name,coordinate_contract='publisher_YOLO_normalized_cxcywh_6th_column_distance_m',frame_sampling_step=STRIDE,
   source_terms={'record':'https://www.macvi.org/workshop/macvi25/challenges/usv_dist','use':'MaCVi 2025 challenge training data; research use'},
   checks_pending=r['checks_pending']+['unlabelled_vessels_full_frame_review'])
  rows=[l.split() for l in lb.decode().splitlines() if l.strip()]
  tasks.append((r,rows))
 def finish(t):
  r,rows=t;r=check_image(r)
  w,h=r['image'].get('width'),r['image'].get('height')
  if w:
   for c,cx,cy,bw,bh,*dist in rows:
    if c!='0':r['issues'].append('unknown_native_class');continue
    cx,cy,bw,bh=float(cx)*w,float(cy)*h,float(bw)*w,float(bh)*h
    x1,y1,x2,y2=max(0.0,cx-bw/2),max(0.0,cy-bh/2),min(float(w),cx+bw/2),min(float(h),cy+bh/2)
    if x2-x1<1 or y2-y1<1:r['issues'].append('degenerate_native_box');continue
    r['boxes'].append({'class_id':3,'xyxy':[round(x1,3),round(y1,3),round(x2,3),round(y2,3)],'native_class':'nav_buoy','native_distance_m':float(dist[0]) if dist else None})
  if not r['boxes']:r['issues'].append('no_native_positive_label_full_frame_review_needed')
  r['issues']=sorted(set(r['issues']));return r
 with ThreadPoolExecutor(8) as ex:staged=list(ex.map(finish,tasks))
 out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in staged),encoding='utf8')
 rep={'state':'CANDIDATES_ONLY','records':len(staged),'recordings':rec+1,'stride':STRIDE,'with_issues':sum(bool(r['issues']) for r in staged),'boxes':sum(len(r['boxes']) for r in staged)}
 write_json(ROOT/'reports/macvi_staging.json',rep);print(json.dumps(rep))
if __name__=='__main__':main()
