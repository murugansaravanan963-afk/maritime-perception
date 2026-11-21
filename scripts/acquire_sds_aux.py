#!/usr/bin/env python3
"""Acquire official SDS TRAIN auxiliary frames, excluding every ambiguous image.
This does not admit labels or turn aerial frames into surface-camera coverage.
"""
import argparse,importlib.util,json,hashlib,shutil,time
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from collections import defaultdict,Counter
from audit_candidates import ROOT,write_json,image_info,sha
OLD=Path(r'D:\Projects\maritime-perception')
# Pin the inspected publisher-only transport implementation before importing it.
TRANSPORT=OLD/'scripts/download_seadronessee.py'
def load_transport():
 expected=json.loads((ROOT/'configs/sds_transport_pin.json').read_text())['sha256']
 if sha(TRANSPORT.read_bytes())!=expected:raise ValueError('Inspected SDS transport changed')
 spec=importlib.util.spec_from_file_location('original_sds_transport',TRANSPORT);mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod);return mod

def main():
 p=argparse.ArgumentParser();p.add_argument('--download',action='store_true');p.add_argument('--workers',type=int,default=6);a=p.parse_args()
 mod=load_transport();p=OLD/'data/SeaDronesSee_v4/annotations/instances_train.json';ab=p.read_bytes();doc=json.loads(ab);cats={x['id']:x['name'] for x in doc['categories']}
 if cats!={0:'ignored',1:'swimmer',2:'boat',3:'jetski',4:'life_saving_appliances',5:'buoy'}:raise ValueError('Unexpected native taxonomy')
 by=defaultdict(list)
 for ann in doc['annotations']:by[ann['image_id']].append(ann)
 selected=[]
 for im in doc['images']:
  aa=by[im['id']]
  if not aa or not im.get('source',{}).get('video'):continue
  if any(x['category_id'] in [0,2] or x.get('iscrowd') or x.get('ignore') for x in aa):continue
  mod.safe_name(im['file_name']);selected.append(im)
 # Only one publisher train directory is inventoried. No val/test image payload.
 metadata={r['name']:r for r in mod.listing('Compressed Version/images/train') if not r['directory']}
 if any(im['file_name'] not in metadata for im in selected):raise ValueError('Selected image absent from publisher train listing')
 plan={'state':'DOWNLOAD_PLAN_AERIAL_AUXILIARY','source_annotation_sha256':sha(ab),'selected_images':len(selected),'bytes':sum(metadata[i['file_name']]['bytes'] for i in selected),'source_videos':len({i['source']['video'] for i in selected}),'native_class_positive_images':dict(Counter(cats[c] for im in selected for c in {x['category_id'] for x in by[im['id']]})),'publisher_test_payloads_opened':0,'primary_coverage_credit':0}
 write_json(ROOT/'reports/sds_aux_acquisition_plan.json',plan);print(json.dumps(plan,indent=2),flush=True)
 if not a.download:return
 out=ROOT/'raw/seadronessee_aux';out.mkdir(parents=True,exist_ok=True)
 (out/'instances_train_original.json').write_bytes(ab)
 keep={im['id'] for im in selected};subset={**doc,'images':selected,'annotations':[x for x in doc['annotations'] if x['image_id'] in keep]}
 write_json(out/'selected_train.json',subset);write_json(out/'publisher_train_listing.json',metadata)
 receiptpath=out/'download_receipts.json';previous=json.loads(receiptpath.read_text()) if receiptpath.exists() else {};receipts=dict(previous);failures=[];start=time.monotonic()
 def one(im):
  fn=im['file_name'];remote='Compressed Version/images/train/'+fn;target=out/'images/train'/fn
  rec=mod.download_file(remote,target,metadata[fn],previous.get(fn));info=image_info(target.read_bytes())
  if (info['width'],info['height'])!=(im['width'],im['height']):raise ValueError('Publisher image dimension mismatch '+fn)
  return fn,{**rec,**info,'native_image_id':im['id'],'video':im['source']['video']}
 with ThreadPoolExecutor(max_workers=a.workers) as ex:
  jobs={ex.submit(one,im):im['file_name'] for im in selected}
  for n,future in enumerate(as_completed(jobs),1):
   fn=jobs[future]
   try:fn,r=future.result();receipts[fn]=r
   except Exception as e:failures.append({'file_name':fn,'error':str(e)})
   if n%50==0 or n==len(jobs):
    write_json(receiptpath,receipts);print('SDS auxiliary',n,'/',len(jobs),'verified',len(receipts),'failed',len(failures),'seconds',round(time.monotonic()-start),flush=True)
 result={**plan,'state':'ACQUIRED_CANDIDATES_NOT_RELEASED' if not failures else 'PARTIAL_ACQUISITION','downloaded_verified':len(receipts),'failures':failures,'admitted_images':0,'bytes_downloaded':sum(r['bytes'] for r in receipts.values())};write_json(ROOT/'reports/sds_aux_acquisition.json',result)
 if failures:raise SystemExit(2)
if __name__=='__main__':main()
