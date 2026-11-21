#!/usr/bin/env python3
"""Fail-closed review and release gate for canonical maritime6 manifests.

This does not certify annotations by model confidence. Reviews must bind the exact
image, source annotation, canonical record and taxonomy examined by a reviewer.
"""
import argparse,hashlib,json,math,shutil,tempfile
from collections import Counter,defaultdict
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
NAMES=['powered_vessel','sailboat','small_craft','navigation_marker','person_in_water','other_obstacle']
def digest(b):return hashlib.sha256(b).hexdigest()
def canonical_hash(record):return digest(json.dumps(record,sort_keys=True,separators=(',',':'),ensure_ascii=False,allow_nan=False).encode())
def read_jsonl(p):
 if not p.exists():return []
 return [json.loads(x) for x in p.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
def gate(records,reviews,policy,taxonomy_sha256,verify_files=True):
 errors=[];accepted=[];seen_ids=set();files={};pixels={};units={};groups={};near={};review_by={}
 for v in reviews:
  rid=v.get('id')
  if rid in review_by:errors.append(f'duplicate_review:{rid}')
  review_by[rid]=v
 counts={s:Counter() for s in ['train','val','test']};totalcounts={s:Counter() for s in counts};groupcounts={s:defaultdict(Counter) for s in counts}
 for r in records:
  rid=r.get('id','<missing>');before=len(errors)
  def err(msg):errors.append(str(rid)+':'+msg)
  if rid in seen_ids:err('duplicate_id')
  seen_ids.add(rid)
  split=r.get('split');source_split=r.get('source_split');view=r.get('viewpoint');group=r.get('capture_group');unit=r.get('independent_unit');cluster=r.get('duplicate_cluster')
  if split not in counts:err('invalid_split');continue
  if source_split not in policy['allowed_source_splits'][split]:err('publisher_split_violation')
  if not r.get('source') or not r.get('source_revision') or not r.get('source_annotation_sha256'):err('missing_source_provenance')
  if not group or not unit or not cluster:err('missing_capture_or_duplicate_group')
  for name,val,table in [('capture_group',group,groups),('independent_unit',unit,units),('duplicate_cluster',cluster,near)]:
   if val in table and table[val]!=split:err(name+'_crosses_splits')
   table[val]=split
  if r.get('issues'):err('unresolved_audit_issues')
  if r.get('unresolved_objects'):err('unresolved_native_objects')
  if r.get('checks_pending'):err('pending_dataset_checks')
  if r.get('augmented_from') or r.get('is_augmented'):err('preaugmented_image_not_eligible_for_release')
  if view not in ['boat','shore','surface','aerial']:err('unknown_viewpoint')
  if view=='surface' and (r.get('viewpoint_basis',{}).get('surface_camera_confirmed') is not True or not r.get('viewpoint_basis',{}).get('reason')):err('missing_surface_viewpoint_evidence')
  review=review_by.get(rid,{})
  try:rh=canonical_hash(r)
  except (ValueError,TypeError):rh=None;err('invalid_canonical_record')
  for k,value in [('record_sha256',rh),('taxonomy_sha256',taxonomy_sha256)]:
   if review.get(k)!=value or value is None:err('review_'+k+'_mismatch')
  for k in ['approved','full_frame_complete','class_and_box_reviewed','capture_group_verified','near_duplicate_audit_complete','source_terms_verified']:
   if review.get(k) is not True:err('missing_review_'+k)
  if not review.get('reviewed_by') or not review.get('reviewed_at'):err('missing_reviewer_identity_or_time')
  im=r.get('image',{});w=im.get('width',0);h=im.get('height',0)
  if not isinstance(w,int) or not isinstance(h,int) or w<=0 or h<=0:err('invalid_dimensions');w=h=0
  hashes={k:im.get(k) for k in ['sha256','pixel_sha256']}
  if not all(isinstance(x,str) and len(x)==64 and all(c in '0123456789abcdef' for c in x) for x in hashes.values()):err('invalid_image_hash')
  if verify_files:
   try:
    p=Path(im['path']);data=p.read_bytes()
    if digest(data)!=im.get('sha256'):err('image_hash_mismatch')
    with Image.open(p) as img:img.verify()
    with Image.open(p) as img:
     img.load();rgb=img.convert('RGB')
     if img.size!=(w,h):err('image_dimension_mismatch')
     if digest(str(rgb.size).encode()+rgb.tobytes())!=im.get('pixel_sha256'):err('decoded_pixel_hash_mismatch')
   except Exception as e:err('image_read_failure:'+type(e).__name__)
  for k,table in [('sha256',files),('pixel_sha256',pixels)]:
   v=im.get(k)
   if v in table:err('duplicate_image_'+k)
   table[v]=rid
  boxes=r.get('boxes');classes=set();seenboxes=set()
  if not isinstance(boxes,list):err('boxes_not_a_list');boxes=[]
  for b in boxes:
   c=b.get('class_id');xy=b.get('xyxy')
   if type(c) is not int or c not in range(6):err('invalid_class');continue
   if not isinstance(xy,list) or len(xy)!=4 or not all(type(x) in [int,float] and math.isfinite(x) for x in xy):err('invalid_box');continue
   x1,y1,x2,y2=xy
   if not(0<=x1<x2<=w and 0<=y1<y2<=h):err('box_outside_image_or_degenerate')
   key=(c,*xy)
   if key in seenboxes:err('duplicate_box')
   seenboxes.add(key);classes.add(c)
  if len(errors)==before:
   accepted.append(r)
   for c in classes:totalcounts[split][c]+=1
   if view in policy['primary_viewpoints']:
    for c in classes:counts[split][c]+=1;groupcounts[split][c][group]+=1
 for s in counts:
  quota=policy[f'min_primary_{s}_images_per_class'];ng=policy['min_train_capture_groups_per_class' if s=='train' else 'min_eval_capture_groups_per_class']
  for c in range(6):
   if counts[s][c]<quota:errors.append(f'coverage:{s}:{NAMES[c]}:{counts[s][c]}<{quota}')
   if len(groupcounts[s][c])<ng:errors.append(f'capture_diversity:{s}:{NAMES[c]}')
   if s=='train' and counts[s][c] and max(groupcounts[s][c].values())/counts[s][c]>policy['max_fraction_of_class_from_one_train_group']:errors.append('one_recording_dominates:'+NAMES[c])
 for c in range(6):
  required=policy.get('min_train_images_per_class',0)
  if totalcounts['train'][c]<required:errors.append(f"total_training_coverage:{NAMES[c]}:{totalcounts['train'][c]}<{required}")
 train_records=[r for r in accepted if r['split']=='train']
 primary_fraction=sum(r['viewpoint'] in policy['primary_viewpoints'] for r in train_records)/len(train_records) if train_records else 0
 if primary_fraction<policy.get('min_primary_fraction_of_training_images',0):errors.append('insufficient_primary_training_image_fraction')
 vals=[totalcounts['train'][c] for c in range(6)]
 if min(vals)>0 and max(vals)/min(vals)>policy['max_train_class_image_ratio']:errors.append('training_class_image_imbalance')
 # Count each independent capture unit only once, even inside one split.
 u=Counter(r.get('independent_unit') for r in accepted)
 if any(n>1 for n in u.values()):errors.append('repeated_independent_unit')
 return {'passed':not errors,'state':'READY_FOR_EXPORT' if not errors else 'NOT_RELEASED','records':len(records),'reviewed_structurally_valid_records':len(accepted),'primary_images_per_class':{s:{NAMES[c]:counts[s][c] for c in range(6)} for s in counts},'total_images_per_class':{s:{NAMES[c]:totalcounts[s][c] for c in range(6)} for s in counts},'primary_training_image_fraction':primary_fraction,'errors':errors}
def publish(records,reviews,policy,taxonomy_bytes,destination):
 if json.loads(taxonomy_bytes).get('names')!=NAMES:raise ValueError('Taxonomy class order mismatch')
 result=gate(records,reviews,policy,digest(taxonomy_bytes))
 if not result['passed']:raise ValueError('Release gate failed; no dataset exported: '+str(result['errors'][:5]))
 if destination.exists():raise FileExistsError('Release path already exists')
 destination.parent.mkdir(parents=True,exist_ok=True)
 # Staging remains available for diagnosis if export fails. Destination is only
 # renamed into place after all image and YOLO files have been written.
 temp=Path(tempfile.mkdtemp(prefix='.building-',dir=destination.parent))
 for r in records:
  stem=digest(r['id'].encode())[:32];s=r['split'];img=r['image'];suffix=Path(img['path']).suffix.lower()
  ip=temp/'images'/s/(stem+suffix);lp=temp/'labels'/s/(stem+'.txt');ip.parent.mkdir(parents=True,exist_ok=True);lp.parent.mkdir(parents=True,exist_ok=True)
  shutil.copy2(img['path'],ip)
  if digest(ip.read_bytes())!=img['sha256']:raise ValueError('Copied image changed')
  w,h=img['width'],img['height'];lines=[]
  for b in r['boxes']:
   x1,y1,x2,y2=b['xyxy'];lines.append(f"{b['class_id']} {(x1+x2)/2/w:.10f} {(y1+y2)/2/h:.10f} {(x2-x1)/w:.10f} {(y2-y1)/h:.10f}")
  lp.write_text('\n'.join(lines)+ ('\n' if lines else ''))
 yaml='path: '+json.dumps(str(destination.resolve()))+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(NAMES))
 (temp/'dataset.yaml').write_text(yaml);(temp/'taxonomy.json').write_bytes(taxonomy_bytes)
 (temp/'manifest.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in records));(temp/'reviews.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in reviews));(temp/'audit.json').write_text(json.dumps(result,indent=2))
 (temp/'release_policy.json').write_text(json.dumps(policy,sort_keys=True,indent=2)+'\n',encoding='utf8')
 metadata_files=['manifest.jsonl','reviews.jsonl','taxonomy.json','release_policy.json','audit.json','dataset.yaml']
 fingerprint={name:digest((temp/name).read_bytes()) for name in metadata_files}
 (temp/'release_fingerprint.json').write_text(json.dumps({'schema':'maritime6-reviewed-release-1','files_sha256':fingerprint,'records':len(records)},indent=2)+'\n',encoding='utf8')
 temp.rename(destination);return result

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--reviews',type=Path,required=True);p.add_argument('--out',type=Path,required=True);p.add_argument('--publish',type=Path);a=p.parse_args()
 if not a.manifest.exists() or not a.reviews.exists():raise FileNotFoundError('Canonical manifest and review ledger must both exist')
 records=read_jsonl(a.manifest);reviews=read_jsonl(a.reviews);policy=json.loads((ROOT/'configs/release_policy.json').read_text());tax=(ROOT/'configs/taxonomy.json').read_bytes()
 result=gate(records,reviews,policy,digest(tax));a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
 if a.publish:publish(records,reviews,policy,tax,a.publish)
 raise SystemExit(0 if result['passed'] else 2)
if __name__=='__main__':main()
