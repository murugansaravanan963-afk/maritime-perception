#!/usr/bin/env python3
"""Verify a frozen release after transfer, using exported files rather than D: paths."""
import argparse,json,math
from pathlib import Path
from release_gate import NAMES,gate,digest,read_jsonl

def verify_release(root):
 root=Path(root).resolve()
 if not (root/'release_fingerprint.json').is_file():raise ValueError('No validated dataset release at '+str(root)+'; candidate inventories cannot be trained')
 fp=json.loads((root/'release_fingerprint.json').read_text(encoding='utf8'))
 pilot=fp.get('schema')=='maritime6-pilot-release-0'
 if fp.get('schema')!='maritime6-reviewed-release-1' and not pilot:raise ValueError('Unknown release schema')
 required={'manifest.jsonl','taxonomy.json','release_policy.json','audit.json','dataset.yaml'}|(set() if pilot else {'reviews.jsonl'})
 if set(fp.get('files_sha256',{}))!=required:raise ValueError('Incomplete release fingerprint')
 for name,h in fp['files_sha256'].items():
  if digest((root/name).read_bytes())!=h:raise ValueError('Changed release metadata: '+name)
 records=read_jsonl(root/'manifest.jsonl');tax=(root/'taxonomy.json').read_bytes();policy=json.loads((root/'release_policy.json').read_text())
 if json.loads(tax)['names']!=NAMES:raise ValueError('Class order changed')
 stored=json.loads((root/'audit.json').read_text())
 if pilot:
  # Pilot releases carry no reviewer attestation ledger; recompute their split/leakage checks instead.
  from pilot_release import check
  result=check(records,policy,verify_files=False);result['dropped_train_only_records_in_holdout_units']=stored.get('dropped_train_only_records_in_holdout_units')
  result['primary_images_per_class']=result['images_containing_class']
 else:result=gate(records,read_jsonl(root/'reviews.jsonl'),policy,digest(tax),verify_files=False)
 if not result['passed'] or len(records)!=fp['records']:raise ValueError('Frozen release gate failed: '+str(result['errors'][:8]))
 if pilot:stored['primary_images_per_class']=stored['images_containing_class']
 if stored!=result:raise ValueError('Frozen audit differs from recomputed metadata audit')
 expected_files=set();labels_seen=0
 from audit_candidates import image_info
 for r in records:
  im=r['image'];stem=digest(r['id'].encode())[:32];split=r['split'];export=im.get('export') if pilot else None
  suffix=export['suffix'] if export else Path(im['path']).suffix.lower()
  ip=root/'images'/split/(stem+suffix);lp=root/'labels'/split/(stem+'.txt');expected_files.update([ip,lp])
  data=ip.read_bytes();info=image_info(data)
  # A pixel-identical lossless re-export has its own file hash; decoded pixels must still match the review.
  if info['image_sha256']!=(export['sha256'] if export else im['sha256']) or info['pixel_sha256']!=im['pixel_sha256'] or (info['width'],info['height'])!=(im['width'],im['height']):raise ValueError('Exported image differs from reviewed image: '+r['id'])
  lines=[x.split() for x in lp.read_text(encoding='utf-8-sig').splitlines() if x.strip()]
  if len(lines)!=len(r['boxes']):raise ValueError('Exported object count changed: '+r['id'])
  for fields,b in zip(lines,r['boxes']):
   if len(fields)!=5:raise ValueError('Malformed exported YOLO label')
   vals=list(map(float,fields));c,*xy=vals
   if not all(math.isfinite(v) for v in vals) or c!=b['class_id']:raise ValueError('Exported class changed')
   x1,y1,x2,y2=b['xyxy'];want=[(x1+x2)/2/im['width'],(y1+y2)/2/im['height'],(x2-x1)/im['width'],(y2-y1)/im['height']]
   if any(abs(a-v)>6e-11 for a,v in zip(xy,want)):raise ValueError('Exported box differs from reviewed box: '+r['id'])
   labels_seen+=1
 actual={p for base in ['images','labels'] for p in (root/base).rglob('*') if p.is_file()}
 # Ultralytics cache files live beside labels; allow caches but no extra training images/labels.
 actual={p for p in actual if p.suffix!='.cache'}
 if actual!=expected_files:raise ValueError('Unexpected/missing image or label files in release')
 return {'state':'VERIFIED_TRANSFERRED_RELEASE','release':str(root),'release_fingerprint_sha256':digest((root/'release_fingerprint.json').read_bytes()),'images_verified':len(records),'boxes_verified':labels_seen,'primary_images_per_class':result['primary_images_per_class']}

def runtime_yaml(release,destination):
 # Original immutable dataset.yaml may contain a Windows path; never edit it.
 root=Path(release).resolve();destination=Path(destination);destination.parent.mkdir(parents=True,exist_ok=True)
 body='path: '+json.dumps(str(root))+'\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n'+''.join(f'  {i}: {n}\n' for i,n in enumerate(NAMES))
 destination.write_text(body,encoding='utf8');return destination

def main():
 p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--runtime-yaml',type=Path);a=p.parse_args();r=verify_release(a.release)
 if a.runtime_yaml:r['runtime_yaml']=str(runtime_yaml(a.release,a.runtime_yaml))
 print(json.dumps(r,indent=2))
if __name__=='__main__':main()
