#!/usr/bin/env python3
"""Inventory official Open Images TRAIN boxes for canoe/surfboard candidates.
Keep all supplied co-occurring classes; do not treat absent target annotations as
proof of background or translate generic Person/Boat to maritime classes.
"""
import argparse,csv,json,hashlib,urllib.request,io,time
from pathlib import Path
from collections import Counter,defaultdict
from PIL import Image
from audit_candidates import ROOT,image_info,write_json
TARGETS={'/m/0ph39':'Canoe','/m/019w40':'Surfboard'}
def inventory(path):
 counts=Counter();images=defaultdict(set);flags=Counter();total=0
 with path.open('rb') as f:
  header=f.readline().decode().strip().split(',')
  if header[0]!='ImageID' or header[2]!='LabelName':raise ValueError('Unexpected CSV schema')
  for line in f:
   total+=1
   # Match native class token with delimiters, never a substring class guess.
   if not any(b','+k.encode()+b',' in line for k in TARGETS):continue
   row=dict(zip(header,next(csv.reader([line.decode()]))));c=row['LabelName'];counts[c]+=1;images[c].add(row['ImageID'])
   for flag in ['IsGroupOf','IsDepiction','IsInside','IsTruncated','IsOccluded']:
    if row.get(flag)=='1':flags[TARGETS[c]+':'+flag]+=1
   if total%1_000_000==0:print(total,'rows checked',flush=True)
 wanted=set().union(*images.values());allanns=defaultdict(list)
 with path.open('rb') as f:
  f.readline()
  for line in f:
   imageid=line.split(b',',1)[0].decode()
   if imageid in wanted:
    row=dict(zip(header,next(csv.reader([line.decode()]))));allanns[imageid].append(row)
 out=ROOT/'reports/openimages_candidate_native_annotations.jsonl'
 with out.open('w') as f:
  for id in sorted(wanted):f.write(json.dumps({'id':id,'source_split':'train','native_annotations':allanns[id]})+'\n')
 report={'source':'Open Images v7 box release using v6 training CSV','native_rows_scanned':total,'native_boxes':{TARGETS[c]:counts[c] for c in TARGETS},'native_images':{TARGETS[c]:len(images[c]) for c in TARGETS},'native_flagged_box_counts':dict(flags),'admitted_images':0,'test_payloads_opened':0,'warning':'Source class inventory only; Canoe and Surfboard need on-water context. All six maritime targets must be reviewed across each full image. Boxes use native normalized xyxy; extent convention still needs harmonization; group/depiction annotations are not individual target boxes.'}
 write_json(ROOT/'reports/openimages_train_native_audit.json',report);print(json.dumps(report,indent=2));return images,allanns

def samples(images,annotations,n):
 # Official downloader identifies this CVDF bucket; no third-party relabelled mirror.
 out=ROOT/'raw/openimages/samples';out.mkdir(parents=True,exist_ok=True);ids=sorted(images.get('/m/0ph39',[]));selected=[ids[int(i*(len(ids)-1)/max(n-1,1))] for i in range(min(n,len(ids)))];receipt=[]
 for id in selected:
  url='https://open-images-dataset.s3.amazonaws.com/train/'+id+'.jpg'
  try:
   p=out/(id+'.jpg')
   if p.exists():data=p.read_bytes()
   else:
    with urllib.request.urlopen(url,timeout=30) as resp:data=resp.read()
    p.write_bytes(data)
   info=image_info(data);receipt.append({'id':id,'url':url,'path':str(p),'image':info,'native_annotations':annotations[id],'state':'unreviewed_candidate'})
  except Exception as e:receipt.append({'id':id,'url':url,'error':str(e),'state':'download_or_decode_failed'})
 write_json(out/'receipts.json',receipt);print('Sample images',sum(x['state']=='unreviewed_candidate' for x in receipt))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--boxes',type=Path,default=ROOT/'raw/openimages/oidv6-train-annotations-bbox.csv');p.add_argument('--samples',type=int,default=0);a=p.parse_args();images,anns=inventory(a.boxes)
 if a.samples:samples(images,anns,a.samples)
