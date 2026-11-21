#!/usr/bin/env python3
"""Acquire original people-context photos; no class is mapped to swimmer here."""
import json,concurrent.futures
from collections import defaultdict
from audit_candidates import ROOT,write_json
from acquire_openimages_canoes import acquire

def main():
 rows=[json.loads(l) for l in (ROOT/'reports/openimages_people_native_candidates.jsonl').read_text(encoding='utf8').splitlines()]
 provenance={r['image_id']:r['provenance'] for r in rows};out=ROOT/'raw/openimages/water_people_photos';out.mkdir(parents=True,exist_ok=True);results=[]
 # Cache verified original sample or canoe images without fetching identical bytes again.
 for r in rows:
  fn=r['image_id']+'.jpg';p=out/fn
  if not p.exists():
   for folder in ['water_people_samples','canoe_photos']:
    candidate=ROOT/'raw/openimages'/folder/fn
    if candidate.is_file():p.write_bytes(candidate.read_bytes());break
 with (ROOT/'reports/openimages_people_photo_manifest.jsonl').open('w',encoding='utf8') as f,concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
  jobs=[pool.submit(acquire,{'id':r['image_id'],'native_annotations':r['native_annotations']},provenance,out) for r in rows]
  for job in concurrent.futures.as_completed(jobs):
   r=job.result();results.append(r);f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush()
   if len(results)%100==0:print('Original people-context candidates',len(results),'/',len(rows),'decoded',sum('pixel_sha256' in x for x in results),flush=True)
 pixels=defaultdict(list)
 for r in results:
  if 'pixel_sha256' in r:pixels[r['pixel_sha256']].append(r['id'])
 report={'state':'ACQUIRED_CONTEXT_CANDIDATES_NOT_SWIMMER_GROUNDTRUTH','selected_images':len(rows),'downloaded_decoded':sum('pixel_sha256' in r for r in results),'failures':[{'id':r['id'],'issues':r['issues']} for r in results if r['state']=='download_or_decode_failed'],'native_geometry_or_provenance_issue_images':sum(bool(r['issues']) for r in results),'unique_decoded_images':len(pixels),'duplicate_pixel_groups':[v for v in pixels.values() if len(v)>1],'bytes':sum(r.get('bytes',0) for r in results),'confirmed_person_in_water_images':0,'admitted_images':0,'context_sample_review':'16 original evenly spaced samples inspected: clear open-water/wading candidates in 5 samples, plus dry-land, poolside and aboard cases. This is source screening, not exhaustive label review; no 5/16 extrapolation credited to coverage.','required_review':['physical in-water status vs dry shore/aboard/pool','all visible six-class targets and full visible extent','source terms/attribution','capture/site/near-duplicate groups']}
 write_json(ROOT/'reports/openimages_people_photo_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k!='required_review'},indent=2))
 if report['failures']:raise SystemExit(2)
if __name__=='__main__':main()
