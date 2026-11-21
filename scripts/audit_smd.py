#!/usr/bin/env python3
"""Audit original SMD ObjectGT metadata without guessing stale text labels."""
import argparse,json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
import numpy as np
from scipy.io import loadmat
ROOT=Path(__file__).resolve().parents[1]
NATIVE={1:'Ferry',2:'Buoy',3:'Vessel/ship',4:'Speed boat',5:'Boat',6:'Kayak',7:'Sail boat',8:'Swimming person',9:'Flying bird/plane',10:'Other'}
def frame_inventory(frame):
 ids=np.asarray(frame.get('Object',[])).ravel();text=np.asarray(frame.get('ObjectType',[]),dtype=object).ravel();boxes=np.asarray(frame.get('BB',[]),dtype=float);issues=[]
 if boxes.size%4:issues.append('bbox_shape');boxes=np.empty((0,4))
 else:boxes=boxes.reshape(-1,4)
 if len(ids)!=len(boxes):issues.append('numeric_id_bbox_count_mismatch')
 if len(text)!=len(ids):issues.append('text_numeric_count_mismatch')
 else:
  for c,n in zip(ids,text):
   if int(c) not in NATIVE or str(n).strip()!=NATIVE.get(int(c)):issues.append('unknown_or_conflicting_native_class');break
 if any(int(c) not in NATIVE for c in ids):issues.append('unknown_native_class_id')
 if not np.isfinite(boxes).all() or any(w<=0 or h<=0 for x,y,w,h in boxes):issues.append('nonfinite_or_degenerate_box')
 return [int(c) for c in ids],boxes,sorted(set(issues))
def audit(folder,out):
 fc=Counter();bc=Counter();clean=Counter();reasons=Counter();per=[];rows=[];mapping=defaultdict(Counter)
 for p in sorted(folder.glob('*ObjectGT.mat')):
  data=p.read_bytes();frames=loadmat(p,simplify_cells=True)['structXML'];frames=[frames] if isinstance(frames,dict) else list(frames);vf=Counter();vc=Counter();bad=Counter()
  for idx,frame in enumerate(frames):
   ids,boxes,issues=frame_inventory(frame);names=[NATIVE.get(c,'UNKNOWN_'+str(c)) for c in ids];fc.update(set(names));bc.update(names);vf.update(set(names));reasons.update(issues);bad.update(issues)
   if not issues:clean.update(set(names));vc.update(set(names))
   if 'text_numeric_count_mismatch' not in issues:
    for c,n in zip(ids,np.asarray(frame.get('ObjectType',[]),dtype=object).ravel()):mapping[c][str(n).strip()]+=1
   rows.append({'video':p.stem.replace('_ObjectGT',''),'mat_frame_index':idx+1,'source_annotation_sha256':hashlib.sha256(data).hexdigest(),'native_class_ids':ids,'native_boxes_xywh':boxes.tolist(),'issues':issues,'status':'unreviewed_candidate' if not issues else 'quarantine'})
  per.append({'video':p.stem.replace('_ObjectGT',''),'frames':len(frames),'images_per_class':dict(vf),'metadata_consistent_images_per_class':dict(vc),'issue_frame_counts':dict(bad)})
 report={'source':'smd_original_shore','videos_with_ObjectGT':len(per),'annotated_frames':len(rows),'native_images_per_class':dict(fc),'native_boxes_per_class':dict(bc),'metadata_consistent_images_per_class':dict(clean),'native_id_name_agreement_evidence':{c:dict(v) for c,v in mapping.items()},'issue_frame_counts':dict(reasons),'per_video':per,'admitted_images':0,'image_payloads_checked':0,'warning':'Text ObjectType vectors are stale in some original frames. Any metadata conflict is quarantined, not guessed. Numeric-ID counts are inventory only; complete six-class image/box review and capture grouping still required.'}
 out.mkdir(parents=True,exist_ok=True);(out/'smd_shore_native_audit.json').write_text(json.dumps(report,indent=2)+'\n')
 with (out/'smd_shore_native_inventory.jsonl').open('w') as f:
  for row in rows:f.write(json.dumps(row)+'\n')
 print(json.dumps({k:v for k,v in report.items() if k not in ['per_video','native_id_name_agreement_evidence']},indent=2))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--folder',type=Path,default=ROOT/'raw/smd_onshore_metadata/VIS_Onshore/ObjectGT');p.add_argument('--out',type=Path,default=ROOT/'reports');a=p.parse_args();audit(a.folder,a.out)
