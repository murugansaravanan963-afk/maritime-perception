"""Acquire original TRAIN photos with all native labels; never approve context."""
import json,concurrent.futures,sys,hashlib
from pathlib import Path
R=Path(__file__).resolve().parents[1];sys.path.insert(0,str(R/'scripts'))
from acquire_openimages_canoes import acquire
from stage_candidates import make_record,check_image
from release_gate import canonical_hash
rows=list(map(json.loads,(R/'reports/openimages_marinecraft_native.jsonl').read_text(encoding='utf8').splitlines()));prov={x['image_id']:x['provenance'] for x in rows};out=R/'raw/openimages/marinecraft_photos';out.mkdir(exist_ok=True)
manifest=R/'reports/openimages_marinecraft_photo_manifest.jsonl';done={}
if manifest.exists():done={r['id'].split(':')[-1]:r for r in map(json.loads,manifest.read_text(encoding='utf8').splitlines())}
with manifest.open('a',encoding='utf8') as dst,concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
 jobs={pool.submit(acquire,{'id':r['image_id'],'native_annotations':r['native_annotations']},prov,out):r['image_id'] for r in rows if r['image_id'] not in done}
 for n,f in enumerate(concurrent.futures.as_completed(jobs),1):
  x=f.result();done[jobs[f]]=x;dst.write(json.dumps(x)+'\n');dst.flush()
  if n%100==0:print(json.dumps({'acquired':len(done),'selected':len(rows),'failed':sum(v['state']=='download_or_decode_failed' for v in done.values())}),flush=True)
records=[]
for k,x in sorted(done.items()):
 if x['state']=='download_or_decode_failed':continue
 ah=canonical_hash(x['native_annotations']);author=x['provenance']['AuthorProfileURL'];r=make_record(x['id'],'openimages_v7',ah,Path(x['image_path']),group='proposal:openimages:author:'+hashlib.sha256(author.encode()).hexdigest()[:24])
 r['source_attribution']=x['provenance'];r['native_annotations']=x['native_annotations'];r['coordinate_contract']='native_OpenImages_normalized_xyxy';r['issues']=x['issues']+['craft_on_water_and_all_frame_targets_unresolved'];r['checks_pending']+=['on_water_vs_stored_or_beached_and_vessel_subtype','all_instances_and_craft','viewpoint_review']
 r['unresolved_objects']=[{'native_label':b['LabelName'],'native_xyxy':[float(b[z])*x['width' if z.startswith('X') else 'height'] for z in ('XMin','YMin','XMax','YMax')]} for b in x['native_annotations']]
 records.append(check_image(r,(x['width'],x['height'])))
p=R/'staging/candidates/openimages_marinecraft.jsonl'
if p.exists():
 old={a['id']:canonical_hash(a) for a in map(json.loads,p.read_text(encoding='utf8').splitlines())}
 for a in records:
  if a['id'] in old and old[a['id']]!=canonical_hash(a):raise ValueError('Existing candidate changed')
p.write_text(''.join(json.dumps(a)+'\n' for a in records),encoding='utf8');print(json.dumps({'staged':len(records),'automatic_approvals':0}),flush=True)
