#!/usr/bin/env python3
"""Acquire original CVDF canoe photo candidates with complete supplied labels.
No image becomes training-approved through this script.
"""
import concurrent.futures,json,urllib.request,time,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from audit_candidates import ROOT,image_info,box_issues

def acquire(row,provenance,out=None):
 id=row['id'];out=Path(out) if out is not None else ROOT/'raw/openimages/canoe_photos';out.mkdir(parents=True,exist_ok=True);p=out/(id+'.jpg');url='https://open-images-dataset.s3.amazonaws.com/train/'+id+'.jpg';rec={'id':'openimages:'+id,'source':'openimages_v7','source_split':'train','url':url,'image_path':str(p),'native_annotations':row['native_annotations'],'provenance':provenance.get(id),'state':'unreviewed_candidate','issues':[]}
 try:
  cache=ROOT/'raw/openimages/samples'/(id+'.jpg')
  if p.exists():data=p.read_bytes()
  elif cache.exists():data=cache.read_bytes();p.write_bytes(data)
  else:
   for attempt in range(3):
    try:
     with urllib.request.urlopen(url,timeout=30) as f:
      if 'image/' not in f.headers.get('Content-Type',''):raise ValueError('Expected image response')
      data=f.read(15_000_001)
      if len(data)>15_000_000:raise ValueError('Image exceeds 15 MB bound')
     break
    except Exception:
     if attempt==2:raise
     time.sleep(attempt+1)
   # Verify full image decode before making the final file visible.
   image_info(data);temp=p.with_suffix('.part');temp.write_bytes(data);temp.replace(p)
  rec.update(image_info(data));rec['bytes']=len(data)
  if not rec['provenance']:rec['issues'].append('missing_source_provenance')
  boxes=[]
  for a in row['native_annotations']:
   xy=[float(a[k]) for k in ['XMin','YMin','XMax','YMax']];boxes.append({'native_class':a['LabelName'],'xyxy':[xy[0]*rec['width'],xy[1]*rec['height'],xy[2]*rec['width'],xy[3]*rec['height']]})
  rec['issues']+=box_issues(boxes,rec['width'],rec['height'])
 except Exception as e:rec['state']='download_or_decode_failed';rec['issues'].append(type(e).__name__+':'+str(e))
 return rec

def eligible_canoe_photo(row):
 cs=[x for x in row['native_annotations'] if x['LabelName']=='/m/0ph39']
 return bool(cs) and not any(x.get('IsDepiction')=='1' or x.get('IsGroupOf')=='1' for x in cs)

def main():
 provenance=json.loads((ROOT/'reports/openimages_candidate_provenance.json').read_text());selected=[]
 for line in (ROOT/'reports/openimages_candidate_native_annotations.jsonl').read_text().splitlines():
  row=json.loads(line)
  if eligible_canoe_photo(row):selected.append(row)
 rows=[];path=ROOT/'reports/openimages_canoe_photo_manifest.jsonl'
 with path.open('w',encoding='utf8') as out,concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  jobs=[pool.submit(acquire,row,provenance) for row in selected]
  for job in concurrent.futures.as_completed(jobs):
   rec=job.result();rows.append(rec);out.write(json.dumps(rec)+'\n');out.flush()
   if len(rows)%100==0:print(len(rows),'/',len(selected),'original canoe photo candidates checked',flush=True)
 pixels=defaultdict(list)
 for x in rows:
  if 'pixel_sha256' in x:pixels[x['pixel_sha256']].append(x['id'])
 report={'source':'original Open Images train / CVDF images','selected_photo_candidates':len(selected),'downloaded_and_decoded':sum('pixel_sha256' in x for x in rows),'download_failures':sum(x['state']=='download_or_decode_failed' for x in rows),'native_geometry_or_provenance_issue_images':sum(bool(x['issues']) for x in rows),'unique_decoded_images':len(pixels),'pixel_duplicate_groups':[v for v in pixels.values() if len(v)>1],'bytes':sum(x.get('bytes',0) for x in rows),'photographer_groups':len({x.get('provenance',{}).get('AuthorProfileURL') for x in rows if x.get('provenance')}),'admitted_images':0,'required_review':['on-water vs stored/off-water','precise canoe/rowboat identity and box extent','all co-occurring six-class targets','photographer/session/near-duplicate grouping','source image reuse terms and attribution']}
 (ROOT/'reports/openimages_canoe_photo_audit.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
