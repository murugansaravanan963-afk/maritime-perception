#!/usr/bin/env python3
"""Inspect original MODD2 metadata and acquire sparse original review frames."""
import io,json,sys,zipfile,hashlib
from pathlib import Path
from collections import defaultdict,Counter
import numpy as np
from scipy.io import loadmat
from inspect_remote_zip import HTTPRangeFile,safe_name
from audit_candidates import ROOT,image_info

def main():
 p=ROOT/'raw/modd2/MODD2_annotations_v2.zip';summary=[];chosen=[];records=[];out=ROOT/'raw/modd2/sparse_samples';out.mkdir(parents=True,exist_ok=True)
 with zipfile.ZipFile(p) as z:
  groups=defaultdict(list)
  for n in z.namelist():
   if n.endswith('L.mat'):groups[n.split('/')[1]].append(n)
  for group,names in sorted(groups.items()):
   pos=[];bad=[];allboxes=0
   for n in sorted(names):
    data=z.read(n);d=loadmat(io.BytesIO(data),simplify_cells=True)['annotations'];b=np.asarray(d.get('obstacles',[]),dtype=float)
    if b.size%4:bad.append(n);continue
    b=b.reshape(-1,4)
    if not np.isfinite(b).all() or any(w<=0 or h<=0 for x,y,w,h in b):bad.append(n);continue
    allboxes+=len(b)
    if len(b):pos.append({'sequence':group,'annotation_member':n,'annotation_sha256':hashlib.sha256(data).hexdigest(),'native_boxes_xywh':b.tolist(),'image_member':'video_data/'+group+'/frames/'+Path(n).stem+'.jpg'})
   summary.append({'sequence':group,'annotation_frames':len(names),'frames_with_obstacle_boxes':len(pos),'native_obstacle_boxes':allboxes,'malformed_or_degenerate_frames':len(bad),'native_semantic_classes':None})
   if pos:chosen.append(pos[len(pos)//2])
  (ROOT/'reports/modd2_native_audit.json').write_text(json.dumps({'sequences':summary,'total_annotation_frames':sum(x['annotation_frames'] for x in summary),'frames_with_obstacle_boxes':sum(x['frames_with_obstacle_boxes'] for x in summary),'native_classes':'large/small relative to water edge, not maritime object classes','admitted_images':0,'warning':'No swimmer count can be inferred from obstacle boxes. Full-frame class and missing-target review is required. Left/right paired frames share the same capture.'},indent=2)+'\n')
 meta=json.loads((ROOT/'research/modd2_raw_video_inventory.json').read_text());s=HTTPRangeFile(meta['url'],meta['archive_bytes'])
 with zipfile.ZipFile(s) as z:
  for i,row in enumerate(chosen):
   n=row['image_member'];safe_name(n);data=z.read(n);row['image']=image_info(data);dest=out/(row['sequence']+'_'+Path(n).name);dest.write_bytes(data);row.update(image_path=str(dest),state='unreviewed_candidate',coordinate_origin='native MATLAB; source convention pending');records.append(row);print('MODD2 original sample',i+1,'/',len(chosen),flush=True)
 (out/'receipts.json').write_text(json.dumps({'source_archive_url':s.url,'source_archive_size':s.size,'range_bytes_downloaded':s.bytes_read,'samples':records},indent=2)+'\n')
 print('Sequences sampled',len(records))
if __name__=='__main__':main()
