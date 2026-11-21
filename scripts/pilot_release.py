#!/usr/bin/env python3
"""Build pilot release r0 from annotation-passed images with a leak-free group split.

Split units = split_guard_group merged through reports/pilot_r0/near_duplicates.json.
Whole units go to one split. Records that may only be trained on (aerial/unknown
views, IALA10 crops) are dropped when their unit is held out, never moved to train.
Unlike release_gate.py this does not require >=1000 training images per class and
does not claim reviewer attestations that were never made; see pilot_release_policy.json.

  python scripts/pilot_release.py --out reports/pilot_r0/split_audit.json
  python scripts/pilot_release.py --out reports/pilot_r0/split_audit.json --publish releases/maritime6_pilot_r0
"""
import argparse,json,math,os,random,shutil,tempfile
from collections import Counter,defaultdict
from pathlib import Path
from release_gate import NAMES,digest,canonical_hash,read_jsonl
ROOT=Path(__file__).resolve().parents[1]
# Working folder holding image_verification.json and near_duplicates.json for this release.
WORK=os.environ.get('PILOT_WORK','reports/pilot_r0')
SCHEMA='maritime6-pilot-release-0'
SPLITS=['train','val','test']

def train_only(r,policy):
 return (r['viewpoint'] in policy['train_only_viewpoints'] or r['source'] in policy['train_only_sources']
         or r.get('split_guard_group') in policy.get('train_only_groups',[]))

def units(records,links):
 parent={}
 def find(x):
  parent.setdefault(x,x)
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 by={r['id']:r for r in records}
 for r in records:find(r['split_guard_group'])
 for l in links:
  if l['a'] in by and l['b'] in by:parent[find(by[l['a']]['split_guard_group'])]=find(by[l['b']]['split_guard_group'])
 return {r['id']:'unit:'+digest(find(r['split_guard_group']).encode())[:16] for r in records}

def assign(records,policy,unit_of):
 """Seeded annealing over whole units toward the val/test fraction of eval-domain positives per class."""
 vec=defaultdict(lambda:[0]*6);drop=Counter();fixed=set();svec=defaultdict(Counter)
 for r in records:
  u=unit_of[r['id']]
  if train_only(r,policy):drop[u]+=1;continue
  for c in {b['class_id'] for b in r['boxes']}:vec[u][c]+=1;svec[u][(c,r['source'])]+=1
 allu=sorted(set(unit_of.values()));fixed={u for u in allu if not any(vec[u])}
 free=[u for u in allu if u not in fixed]
 total=[sum(vec[u][c] for u in free) for c in range(6)]
 target={'val':[policy['val_fraction_target']*t for t in total],'test':[policy['test_fraction_target']*t for t in total]}
 # Optional source balance: val/test should hold every source the model also trains on,
 # so a held-out score is not dominated by one source that training barely contains.
 sb=policy.get('source_balance');stot=Counter()
 for u in free:stot.update(svec[u])
 skeys=[k for k,n in stot.items() if sb and n>=sb['min_source_positives']]
 def cost(state):
  cnt={s:[0]*6 for s in SPLITS};ng={s:[0]*6 for s in SPLITS};d=0;scnt={s:Counter() for s in SPLITS}
  for u in free:
   s=state[u]
   for c in range(6):
    cnt[s][c]+=vec[u][c];ng[s][c]+=vec[u][c]>0
   if skeys:scnt[s].update(svec[u])
   if s!='train':d+=drop[u]
  e=0.0
  for s in ['val','test']:
   for c in range(6):
    e+=((cnt[s][c]-target[s][c])/max(target[s][c],1))**2
    if cnt[s][c]<policy[f'min_primary_{s}_images_per_class']:e+=10
    if ng[s][c]<policy['min_eval_groups_per_class']:e+=10
  for k in skeys:
   n=stot[k];share=n/max(total[k[0]],1)
   if scnt['train'][k]<sb['min_train_share']*n:e+=sb['weight']*share*10
   for s in ['val','test']:
    want=policy[f'{s}_fraction_target']*n;e+=sb['weight']*share*((scnt[s][k]-want)/max(want,1))**2
  return e+d/2000
 rng=random.Random(policy['seed']);best=None
 for restart in range(8):
  state={u:'train' for u in free};cur=cost(state);temp=1.0
  for step in range(30000):
   u=rng.choice(free);old=state[u];state[u]=rng.choice([s for s in SPLITS if s!=old]);new=cost(state)
   if new<=cur or rng.random()<math.exp((cur-new)/max(temp,1e-6)):cur=new
   else:state[u]=old
   temp*=0.9997
  if best is None or cur<best[0]:best=(cur,dict(state))
 split={u:'train' for u in fixed};split.update(best[1]);return split

def check(records,policy,verify_files=True):
 """Fail-closed structural and leakage checks for a pilot manifest."""
 errors=[];seen=set();hashes={'sha256':{},'pixel_sha256':{}};owner={'pilot_unit':{},'split_guard_group':{},'independent_unit':{},'duplicate_cluster':{}}
 # Demo releases split short runs of consecutive frames at random, so a recording may span splits.
 if policy.get('split_mode')=='random_frame_runs':del owner['split_guard_group']
 counts={s:Counter() for s in SPLITS};groups={s:defaultdict(set) for s in SPLITS};images=Counter()
 for r in records:
  rid=r.get('id');s=r.get('split')
  def err(m):errors.append(f'{rid}:{m}')
  if rid in seen:err('duplicate_id')
  seen.add(rid)
  if s not in SPLITS:err('invalid_split');continue
  images[s]+=1
  if r.get('source_split') not in policy['allowed_source_splits'][s]:err('publisher_split_violation')
  if s!='train' and (r.get('viewpoint') not in policy['eval_viewpoints'] or train_only(r,policy)):err('train_only_record_in_holdout')
  for k,table in owner.items():
   v=r.get(k)
   if not v:err('missing_'+k)
   elif table.setdefault(v,s)!=s:err(k+'_crosses_splits')
  if r.get('issues') or r.get('unresolved_objects'):err('unresolved_issues')
  o=r.get('visual_observation') or {}
  if o.get('decision')!='annotation_pass' or o.get('record_sha256')!=r.get('candidate_record_sha256') or o.get('image_sha256')!=r['image'].get('sha256'):err('missing_bound_annotation_pass')
  im=r['image'];w,h=im.get('width'),im.get('height')
  for k in hashes:
   if hashes[k].setdefault(im.get(k),rid)!=rid:err('duplicate_image_'+k)
  if verify_files:
   from PIL import Image
   p=Path(im['path'])
   if digest(p.read_bytes())!=im['sha256']:err('image_hash_mismatch')
   with Image.open(p) as img:
    if img.size!=(w,h):err('image_dimension_mismatch')
  keys=set();classes=set()
  if not r.get('boxes'):err('no_boxes')
  for b in r.get('boxes',[]):
   c=b.get('class_id');xy=b.get('xyxy')
   if type(c) is not int or c not in range(6):err('invalid_class');continue
   if not(isinstance(xy,list) and len(xy)==4 and all(type(x) in (int,float) and math.isfinite(x) for x in xy) and 0<=xy[0]<xy[2]<=w and 0<=xy[1]<xy[3]<=h):err('invalid_box');continue
   k=(c,*xy)
   if k in keys:err('duplicate_box')
   keys.add(k);classes.add(c)
  for c in classes:
   if s=='train' or r['viewpoint'] in policy['eval_viewpoints']:counts[s][c]+=1;groups[s][c].add(r.get('pilot_unit'))
 for s in ['val','test']:
  for c in range(6):
   if counts[s][c]<policy[f'min_primary_{s}_images_per_class']:errors.append(f'coverage:{s}:{NAMES[c]}:{counts[s][c]}')
   if len(groups[s][c])<policy['min_eval_groups_per_class']:errors.append(f'eval_group_diversity:{s}:{NAMES[c]}:{len(groups[s][c])}')
 return {'schema':SCHEMA,'passed':not errors,'records':len(records),'images_per_split':dict(images),
  'images_containing_class':{s:{NAMES[c]:counts[s][c] for c in range(6)} for s in SPLITS},
  'units_containing_class':{s:{NAMES[c]:len(groups[s][c]) for c in range(6)} for s in SPLITS},'errors':errors}

def build(policy):
 records=read_jsonl(ROOT/'staging/annotation_review_pass.jsonl')
 ver=json.loads((ROOT/WORK/'image_verification.json').read_text(encoding='utf8'))
 if ver['manifest_sha256']!=digest((ROOT/'staging/annotation_review_pass.jsonl').read_bytes()):raise ValueError('Image verification is stale; rerun pilot_verify_images.py')
 if ver['failures']:raise ValueError('Image verification failures: '+str(list(ver['failures'])[:5]))
 links=json.loads((ROOT/WORK/'near_duplicates.json').read_text(encoding='utf8'))['links']
 unit_of=units(records,links);split=assign(records,policy,unit_of);out=[];dropped=[]
 for r in records:
  r=dict(r);r['pilot_unit']=unit_of[r['id']];r['split']=split[r['pilot_unit']]
  if r['split']!='train' and train_only(r,policy):dropped.append({'id':r['id'],'unit':r['pilot_unit'],'reason':'train_only_record_in_held_out_unit'});continue
  out.append(r)
 return out,dropped

def publish(records,policy,policy_bytes,result,destination):
 if not result['passed']:raise ValueError('Pilot checks failed; nothing exported: '+str(result['errors'][:5]))
 if destination.exists():raise FileExistsError('Release path already exists')
 tax=(ROOT/'configs/taxonomy.json').read_bytes()
 if json.loads(tax)['names']!=NAMES:raise ValueError('Taxonomy class order mismatch')
 destination.parent.mkdir(parents=True,exist_ok=True);temp=Path(tempfile.mkdtemp(prefix='.building-',dir=destination.parent))
 from PIL import Image
 for r in records:
  stem=digest(r['id'].encode())[:32];s=r['split'];img=r['image'];data=Path(img['path']).read_bytes();suffix=Path(img['path']).suffix.lower()
  lp=temp/'labels'/s/(stem+'.txt');lp.parent.mkdir(parents=True,exist_ok=True);(temp/'images'/s).mkdir(parents=True,exist_ok=True)
  if suffix in ('.jpg','.jpeg') and not data.endswith(b'\xff\xd9'):
   # Ultralytics re-saves such JPEGs in place (lossy), which would alter the reviewed image
   # (and raw/ through the hard link). Export pixel-identical lossless PNG instead.
   ip=temp/'images'/s/(stem+'.png')
   with Image.open(img['path']) as im:im.convert('RGB').save(ip,format='PNG')
   with Image.open(ip) as im:rgb=im.convert('RGB')
   if digest(str(rgb.size).encode()+rgb.tobytes())!=img['pixel_sha256']:raise ValueError('Lossless export changed pixels '+r['id'])
   img['export']={'suffix':'.png','sha256':digest(ip.read_bytes()),'reason':'jpeg_without_trailing_eoi_marker_is_rewritten_by_ultralytics'}
  else:
   ip=temp/'images'/s/(stem+suffix)
   # Hard link on the same volume keeps raw/ immutable without doubling disk use.
   try:os.link(img['path'],ip)
   except OSError:shutil.copy2(img['path'],ip)
   if digest(ip.read_bytes())!=img['sha256']:raise ValueError('Exported image changed '+r['id'])
  w,h=img['width'],img['height']
  lp.write_text(''.join(f"{b['class_id']} {(b['xyxy'][0]+b['xyxy'][2])/2/w:.10f} {(b['xyxy'][1]+b['xyxy'][3])/2/h:.10f} {(b['xyxy'][2]-b['xyxy'][0])/w:.10f} {(b['xyxy'][3]-b['xyxy'][1])/h:.10f}\n" for b in r['boxes']))
 (temp/'dataset.yaml').write_text('path: '+json.dumps(str(destination.resolve()))+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(NAMES)))
 (temp/'taxonomy.json').write_bytes(tax);(temp/'release_policy.json').write_bytes(policy_bytes)
 (temp/'manifest.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in records),encoding='utf8')
 (temp/'audit.json').write_text(json.dumps(result,indent=2)+'\n',encoding='utf8')
 names=['manifest.jsonl','taxonomy.json','release_policy.json','audit.json','dataset.yaml']
 (temp/'release_fingerprint.json').write_text(json.dumps({'schema':SCHEMA,'files_sha256':{n:digest((temp/n).read_bytes()) for n in names},'records':len(records)},indent=2)+'\n',encoding='utf8')
 temp.rename(destination)

def main():
 p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--publish',type=Path);a=p.parse_args()
 pb=(ROOT/os.environ.get('PILOT_POLICY','configs/pilot_release_policy.json')).read_bytes();policy=json.loads(pb)
 records,dropped=build(policy);result=check(records,policy);result['dropped_train_only_records_in_holdout_units']=len(dropped)
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps({**result,'dropped':dropped},indent=2)+'\n',encoding='utf8')
 print(json.dumps({k:v for k,v in result.items() if k!='errors'},indent=2));print('errors:',result['errors'][:20])
 if a.publish:publish(records,policy,pb,result,a.publish)
 raise SystemExit(0 if result['passed'] else 2)
if __name__=='__main__':main()
