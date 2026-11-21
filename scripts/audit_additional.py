#!/usr/bin/env python3
"""Audit acquired original Oostpoort/instream-wood pairs, without admission."""
import argparse,json,zipfile,re,math
from pathlib import Path
from collections import Counter,defaultdict
from audit_candidates import ROOT,sha,image_info,box_issues,write_json

def audit(source):
 if source=='oostpoort':
  iz=zipfile.ZipFile(ROOT/'raw/oostpoort/images.zip');az=zipfile.ZipFile(ROOT/'raw/oostpoort/labels_txt.zip');ims=sorted(n for n in iz.namelist() if Path(n).suffix.lower() in ['.jpg','.png','.jpeg']);anns={Path(n).stem:n for n in az.namelist() if n.endswith('.txt')};split='unspecified'
 else:
  iz=az=zipfile.ZipFile(ROOT/'raw/instream_wood/20201127_rotated.zip');ims=sorted(n for n in iz.namelist() if n.endswith('.jpg'));anns={Path(n).stem:n for n in az.namelist() if n.endswith('.txt')};split='unspecified'
 rows=[]
 for idx,n in enumerate(ims):
  rec={'id':source+':'+Path(n).stem,'source':source,'archive':str(iz.filename),'image_member':n,'source_split':split,'state':'unreviewed_candidate','issues':[]}
  try:
   rec.update(image_info(iz.read(n)));ann=anns[Path(n).stem];data=az.read(ann);boxes=[]
   for line in data.decode('utf-8-sig').splitlines():
    if not line.strip():continue
    parts=line.split()
    if len(parts)!=5:raise ValueError('invalid_yolo_line')
    c,x,y,w,h=map(float,parts)
    if c!=0:raise ValueError('unexpected_native_class')
    boxes.append({'native_class':int(c),'xyxy':[(x-w/2)*rec['width'],(y-h/2)*rec['height'],(x+w/2)*rec['width'],(y+h/2)*rec['height']]})
   rec.update(annotation_archive=str(az.filename),annotation_member=ann,annotation_sha256=sha(data),boxes=boxes,coordinate_contract='native_yolo_normalized_xywh')
   rec['issues']+=box_issues(boxes,rec['width'],rec['height'])
   if source=='oostpoort':
    m=re.match(r'(Oos_exp\d+)',Path(n).stem);rec['group_proposal']=m.group(1) if m else 'unknown'
   else:rec['group_proposal']='instream_wood:20201127' # Conservative date group, includes alternate camera rotations.
  except Exception as e:rec['issues'].append(type(e).__name__+':'+str(e))
  rows.append(rec)
  if (idx+1)%150==0:print(source,idx+1,'images decoded',flush=True)
 with (ROOT/f'reports/{source}_native_manifest.jsonl').open('w',encoding='utf8') as f:
  for rec in rows:f.write(json.dumps(rec)+'\n')
 exact=defaultdict(list)
 for rec in rows:
  if 'pixel_sha256' in rec:exact[rec['pixel_sha256']].append(rec['id'])
 report={'source':source,'state':'AUDIT_ONLY_NOT_RELEASED','images':len(rows),'positive_images':sum(bool(x.get('boxes')) for x in rows),'native_boxes':sum(len(x.get('boxes',[])) for x in rows),'images_with_issues':sum(bool(x['issues']) for x in rows),'unique_decoded_images':len(exact),'pixel_duplicate_groups':[v for v in exact.values() if len(v)>1],'proposed_groups':dict(Counter(x.get('group_proposal','unknown') for x in rows)),'admitted_images':0,'required_next':['confirm class is water-supported external object','verify all six target classes in full frame','recover recording/site group, thin time, cross-source near-duplicate review','allocate independent groups before model selection']}
 write_json(ROOT/f'reports/{source}_native_audit.json',report);print(json.dumps(report));iz.close()
 if az is not iz:az.close()
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('source',choices=['oostpoort','instream_wood']);audit(p.parse_args().source)
