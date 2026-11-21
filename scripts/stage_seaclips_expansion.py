"""Append newly available native TRAIN SeaClips records; never overwrite reviews."""
import json,sys
from pathlib import Path
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor
from stage_candidates import ROOT,OLD,jsonl,make_record,check_image,native_box,MAP_SEACLIPS
from release_gate import canonical_hash
from audit_candidates import sha,write_json

def main():
 source=OLD/'data/SeaClips/seaclips-train.json';data=source.read_bytes();d=json.loads(data);ah=sha(data);by=defaultdict(list)
 for a in d['annotations']:by[a['image_id']].append(a)
 seen={r['id'] for p in (ROOT/'staging/candidates').glob('*.jsonl') for r in jsonl(p)}
 tasks=[]
 for im in d['images']:
  rid='seaclips:'+str(im['id']);aa=by[im['id']]
  if rid in seen or not aa or any(a['category_id'] not in MAP_SEACLIPS or a.get('iscrowd') for a in aa):continue
  r=make_record(rid,'seaclips',ah,OLD/'data/SeaClips/train'/im['file_name'],group='proposal:seaclips:video_'+str(im['video_id']),view='boat')
  r.update(native_image_id=im['id'],native_image_metadata=im,source_annotation_detail_sha256=canonical_hash(aa),coordinate_contract='native_COCO_xywh_exclusive_max',frame_sampling_step=1)
  r['boxes']=[{'class_id':MAP_SEACLIPS[a['category_id']],'xyxy':native_box(a['bbox']),'native_annotation_id':a.get('id')} for a in aa]
  tasks.append((r,(im['width'],im['height'])))
 out=ROOT/'staging/candidates/seaclips_expansion.jsonl';assert not out.exists(),'Refuse overwrite append inventory'
 rows=[]
 with ThreadPoolExecutor(max_workers=4) as pool:
  for n,r in enumerate(pool.map(lambda t:check_image(*t),tasks),1):
   r['issues']=sorted(set(r['issues']));rows.append(r)
   if n%500==0:print(json.dumps({'decoded':n,'selected':len(tasks)}),flush=True)
 out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
 report={'state':'CANDIDATES_ONLY','records':len(rows),'source_annotation_sha256':ah,'geometry_or_decode_holds':sum(bool(r['issues']) for r in rows),'class_positive_counts':dict(Counter(str(c) for r in rows for c in set(b['class_id'] for b in r['boxes']))),'existing_ids_and_hash_bound_reviews_preserved':True,'release_approved':0}
 write_json(ROOT/'reports/seaclips_expansion.json',report);print(json.dumps(report))
if __name__=='__main__':main()
