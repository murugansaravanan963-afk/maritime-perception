#!/usr/bin/env python3
"""Original MOBDrone TRAIN PNGs: exact ZIP members and all five native classes.
No publisher test data acquired. Generic boat frames excluded as a whole.
Native person maps provisionally to water-person; actual full-frame review must
confirm water contact. No decoded/re-encoded video frames used as equivalents.
"""
import argparse,json,hashlib,time,threading,urllib.error,zipfile,zlib
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from remote_zip_member import fetch_member
from inspect_remote_zip import safe_name
from stage_candidates import make_record,check_image
from release_gate import canonical_hash
ROOT=Path(__file__).resolve().parents[1];lock=threading.Lock();next_request=0.
def paced_fetch(index,entry):
 global next_request
 for attempt in range(7):
  with lock:
   delay=max(0,next_request-time.monotonic());next_request=max(next_request,time.monotonic())+1.1
  if delay:time.sleep(delay)
  try:return fetch_member(index['archive_url'],index['archive_bytes'],entry)
  except urllib.error.HTTPError as ex:
   if ex.code not in (429,500,502,503,504):raise
   after=ex.headers.get('Retry-After','90');backoff=max(30,min(int(after) if after.isdigit() else 90,300))
   with lock:next_request=max(next_request,time.monotonic()+backoff)
   print(json.dumps({'publisher_status':ex.code,'global_backoff_seconds':backoff}),flush=True)
  except (TimeoutError,OSError):
   if attempt==6:raise
   time.sleep(5+attempt*3)
 raise RuntimeError('Publisher retries exhausted; member remains unacquired')
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=96);ap.add_argument('--stage-only',action='store_true');a=ap.parse_args()
 base=ROOT/'raw/mobdrone';index=json.loads((base/'remote_archive_index.json').read_text(encoding='utf8'));annpath=base/'annotations_5_custom_classes.json';data=annpath.read_bytes()
 if hashlib.md5(data).hexdigest()!='11f578ff9ede473005d72ccccd0ae82b':raise ValueError('Publisher annotation MD5 mismatch')
 d=json.loads(data);ah=hashlib.sha256(data).hexdigest();by=defaultdict(list)
 for b in d['annotations']:by[b['image_id']].append(b)
 train={i['id']:i for i in d['images'] if i['file_name'].startswith('DJI_0915')};eligible={}
 for k,im in train.items():
  aa=by[k]
  if int(Path(im['file_name']).stem.rsplit('_',1)[1])%30 or not aa or any(b['category_id']==2 or b.get('iscrowd') for b in aa):continue
  if any(b['category_id'] not in (1,3,4,5) for b in aa):raise ValueError('Unknown native category')
  eligible[k]=im
 entries={Path(e['name']).name:e for e in index['entries'] if e['name'].lower().endswith('.png')};keys=sorted(eligible,key=lambda k:hashlib.sha256(('20261023:'+str(k)).encode()).hexdigest())
 if a.limit:keys=keys[:a.limit]
 if a.stage_only:keys=[]
 receiptpath=base/'image_receipts.jsonl';receipts={}
 if receiptpath.exists():receipts={r['native_image_id']:r for r in map(json.loads,receiptpath.read_text(encoding='utf8').splitlines())}
 failures=[]
 def get(k):
  im=eligible[k];e=entries[im['file_name']];p=base.joinpath(*safe_name(e['name']));p.parent.mkdir(parents=True,exist_ok=True)
  if p.exists():payload=p.read_bytes()
  else:
   payload=paced_fetch(index,e);tmp=p.with_suffix('.PNG.part');tmp.write_bytes(payload);tmp.replace(p)
  if len(payload)!=e['bytes'] or format(zlib.crc32(payload)&0xffffffff,'08x')!=e['crc32']:raise ValueError('Original PNG CRC or length mismatch')
  return {'native_image_id':k,'file_name':im['file_name'],'member':e['name'],'path':str(p),'bytes':len(payload),'sha256':hashlib.sha256(payload).hexdigest(),'crc32':e['crc32'],'archive_url':index['archive_url'],'publisher_record':5996890,'source_split':'train'}
 with ThreadPoolExecutor(max_workers=4) as pool:
  jobs={pool.submit(get,k):k for k in keys}
  for n,f in enumerate(as_completed(jobs),1):
   k=jobs[f]
   try:
    r=f.result();receipts[k]=r
    with receiptpath.open('a',encoding='utf8') as out:out.write(json.dumps(r)+'\n')
   except Exception as ex:failures.append({'native_image_id':k,'error':str(ex)});print(json.dumps(failures[-1]),flush=True)
   if n%16==0:print(json.dumps({'processed':n,'selected':len(keys),'failed':len(failures)}),flush=True)
 receiptpath.write_text(''.join(json.dumps(r)+'\n' for k,r in sorted(receipts.items())),encoding='utf8');records=[];mapping={1:4,3:2,4:5,5:5}
 for k,rec in sorted(receipts.items()):
  if k not in eligible:raise ValueError('Noneligible receipt')
  im=eligible[k];aa=by[k];r=make_record('mobdrone:'+str(k),'mobdrone',ah,Path(rec['path']),group='proposal:mobdrone:0915_campaign',view='aerial')
  r['source_revision']='zenodo:5996890:v1.0.0';r['native_image_id']=k;r['native_image_metadata']=im;r['source_annotation_detail_sha256']=canonical_hash(aa);r['native_capture_prefix']=Path(im['file_name']).stem.rsplit('_',1)[0];r['frame_sampling_step']=30;r['coordinate_contract']='publisher_COCO_xywh_exclusive_max';r['source_archive_member']=rec['member']
  r['source_terms']={'license':'CC-BY-4.0','record':'https://zenodo.org/records/5996890'}
  r['boxes']=[{'class_id':mapping[b['category_id']],'xyxy':[b['bbox'][0],b['bbox'][1],b['bbox'][0]+b['bbox'][2],b['bbox'][1]+b['bbox'][3]],'native_annotation_id':b['id'],'native_category_id':b['category_id'],'native_attributes':b.get('attributes',{})} for b in aa]
  r['checks_pending']+=['native_person_physically_in_water_context']
  records.append(check_image(r,(im['width'],im['height'])))
 target=ROOT/'staging/candidates/mobdrone.jsonl'
 if target.exists():
  old={r['id']:r for r in map(json.loads,target.read_text(encoding='utf8').splitlines())}
  if any(r['id'] in old and canonical_hash(old[r['id']])!=canonical_hash(r) for r in records):raise ValueError('Raw candidate changed')
 target.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf8')
 report={'eligible_train_frames_1second_interval':len(eligible),'publisher_train_metadata_images':len(train),'publisher_claim_train_images':88568,'metadata_count_discrepancy':2,'acquired_staged':len(records),'invalid_geometry_or_image_records':sum(bool(r['issues']) for r in records),'failures':failures,'publisher_test_images_acquired':0,'automatic_approvals':0,'person_mapping':'Provisional; all frames require actual water-context review','full_publisher_image_archive_md5_verified':False,'individual_members_crc_and_sha_verified':True}
 (ROOT/'reports/mobdrone_acquisition.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report),flush=True)
 if failures:raise SystemExit(1)
if __name__=='__main__':main()
