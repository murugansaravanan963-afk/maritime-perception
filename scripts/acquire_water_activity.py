import json,concurrent.futures,sys,hashlib
from pathlib import Path
p=Path(r'D:\Projects\maritime-dataset');sys.path.insert(0,str(p/'scripts'))
from acquire_openimages_canoes import acquire
from stage_candidates import make_record,check_image
from release_gate import canonical_hash
rows=list(map(json.loads,(p/'reports/openimages_water_activity_native.jsonl').read_text(encoding='utf8').splitlines()));prov={x['image_id']:x['provenance'] for x in rows};out=p/'raw/openimages/water_activity_photos';out.mkdir(exist_ok=True)
manifest=p/'reports/openimages_water_activity_photo_manifest.jsonl';done={}
if manifest.exists():done={r['id'].split(':')[-1]:r for r in map(json.loads,manifest.read_text(encoding='utf8').splitlines())}
with manifest.open('a',encoding='utf8') as dst,concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 jobs={pool.submit(acquire,{'id':r['image_id'],'native_annotations':r['native_annotations']},prov,out):r['image_id'] for r in rows if r['image_id'] not in done}
 for n,f in enumerate(concurrent.futures.as_completed(jobs),1):
  x=f.result();done[jobs[f]]=x;dst.write(json.dumps(x)+'\n');dst.flush()
  if n%50==0:print(json.dumps({'acquired':len(done),'selected':len(rows),'failed':sum(v['state']=='download_or_decode_failed' for v in done.values())}),flush=True)
records=[]
for k,x in sorted(done.items()):
 if x['state']=='download_or_decode_failed':continue
 ah=canonical_hash(x['native_annotations']);author=x['provenance'].get('AuthorProfileURL','unknown');r=make_record(x['id'],'openimages_v7',ah,Path(x['image_path']),group='proposal:openimages:author:'+hashlib.sha256(author.encode()).hexdigest()[:24])
 r['source_attribution']=x['provenance'];r['native_annotations']=x['native_annotations'];r['coordinate_contract']='native_OpenImages_normalized_xyxy';r['issues']=x['issues']+['person_in_water_status_not_established'];r['checks_pending']+=['in_water_vs_shore_aboard_or_pool','person_visible_body_extent','viewpoint_review']
 r['unresolved_objects']=[{'native_label':b['LabelName'],'native_xyxy':[float(b[z])*x['width' if z.startswith('X') else 'height'] for z in ('XMin','YMin','XMax','YMax')]} for b in x['native_annotations']]
 records.append(check_image(r,(x['width'],x['height'])))
(p/'staging/candidates/openimages_water_activity.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf8');print(json.dumps({'selected':len(rows),'staged':len(records),'automatic_swimmer_mapping':False,'approvals':0}),flush=True)
