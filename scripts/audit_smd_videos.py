#!/usr/bin/env python3
"""Verify original SMD video/GT alignment and stage sparse review frames only."""
import json,hashlib
from pathlib import Path
from collections import Counter
import cv2,numpy as np
from scipy.io import loadmat
from audit_smd import ROOT,frame_inventory,NATIVE

def main():
 folder=ROOT/'raw/smd_shore_videos/VIS_Onshore/Videos';rows=[];stats=[]
 out=ROOT/'staging/smd_sparse_review';out.mkdir(parents=True,exist_ok=True)
 for p in sorted(folder.glob('*.avi')):
  mat=ROOT/'raw/smd_onshore_metadata/VIS_Onshore/ObjectGT'/(p.stem+'_ObjectGT.mat');frames=loadmat(mat,simplify_cells=True)['structXML'];frames=[frames] if isinstance(frames,dict) else list(frames);cap=cv2.VideoCapture(str(p));fps=cap.get(cv2.CAP_PROP_FPS);reported=int(cap.get(cv2.CAP_PROP_FRAME_COUNT));count=0;bad=Counter();candidates=0
  while True:
   ok,img=cap.read()
   if not ok:break
   if count>=len(frames):raise ValueError('More video frames than native annotation frames')
   ids,boxes,issues=frame_inventory(frames[count]);h,w=img.shape[:2]
   for b in boxes:
    x,y,bw,bh=b
    if not (0<=x<x+bw<=w and 0<=y<y+bh<=h):issues.append('native_box_outside_decoded_frame');break
   bad.update(set(issues))
   if 6 in ids and not issues:
    candidates+=1
    # Source-frame IDs stay explicit; sample only every ~one second.
    if count%max(1,round(fps))==0:
     dest=out/(p.stem+f'_frame{count+1:06d}.jpg');cv2.imwrite(str(dest),img,[cv2.IMWRITE_JPEG_QUALITY,98]);rows.append({'source_video':str(p),'source_frame_index_zero_based':count,'source_mat_index_one_based':count+1,'fps':fps,'image_path':str(dest),'width':w,'height':h,'native_class_ids':ids,'native_boxes_xywh':boxes.tolist(),'coordinate_origin':'native MATLAB, conversion convention pending','state':'unreviewed_candidate','capture_group':'SMD_shore_session_unverified'})
   count+=1
  cap.release()
  if count!=len(frames) or count!=reported:raise ValueError('Video/annotation frame count mismatch')
  stats.append({'video':p.name,'decoded_frames':count,'annotation_frames':len(frames),'fps':fps,'geometry_and_metadata_consistent_kayak_frames':candidates,'issue_frame_counts':dict(bad)})
 (out/'native_review_records.json').write_text(json.dumps(rows,indent=2)+'\n');(ROOT/'reports/smd_video_alignment_audit.json').write_text(json.dumps({'videos':stats,'sparse_frames_staged':len(rows),'admitted_images':0,'warning':'Equal frame counts are necessary but not sufficient. Inspect native-frame overlays for temporal alignment and all six target classes before approving.'},indent=2)+'\n');print(json.dumps(stats));print('sparse review images',len(rows))
if __name__=='__main__':main()
