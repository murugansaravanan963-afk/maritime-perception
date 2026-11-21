"""Discover native TRAIN people using water-activity captions, not swimwear.
Captions only prioritize review. No person is automatically labelled in water.
All publisher native boxes retained, existing original image IDs excluded.
"""
import csv,re,json,hashlib,sys
from collections import defaultdict,Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 existing=set()
 for f in (ROOT/'staging/candidates').glob('openimages*.jsonl'):
  for l in f.open(encoding='utf8'):
   r=json.loads(l);existing.add(Path(r['image']['path']).stem)
 metadata={}
 with (ROOT/'raw/openimages/train-images-boxable-with-rotation.csv').open(encoding='utf-8-sig',newline='') as f:
  for r in csv.DictReader(f):
   t=r.get('Title','').lower()
   if r['ImageID'] in existing or not re.search(r'swim|snork|wad[ei]|bath(?:ing)?\b|open.water',t):continue
   if re.search(r'fashion|swimwear|swimsuit|bikini|lingerie|runway|model|pool|bathtub|bathroom|bathhouse|bathrobe|cosplay|suit ',t):continue
   metadata[r['ImageID']]=r
 print(json.dumps({'new_caption_candidates':len(metadata)}),flush=True)
 names=dict(csv.reader((ROOT/'research/openimages_boxable_classes.csv').open(encoding='utf-8-sig')))
 people={k for k,v in names.items() if v in ('Person','Man','Woman','Boy','Girl')};pools={k for k,v in names.items() if v=='Swimming pool'}
 native=defaultdict(list)
 with (ROOT/'raw/openimages/oidv6-train-annotations-bbox.csv').open(encoding='utf-8-sig',newline='') as f:
  for n,r in enumerate(csv.DictReader(f),1):
   if r['ImageID'] in metadata:native[r['ImageID']].append(r)
   if n%3000000==0:print(json.dumps({'native_training_boxes_scanned':n}),flush=True)
 rows=[];stats=Counter()
 for k,aa in native.items():
  pp=[v for v in aa if v['LabelName'] in people]
  if not pp:stats['without_native_person_boxes']+=1;continue
  if any(v['LabelName'] in pools for v in aa):stats['native_pool_excluded']+=1;continue
  if any(v['IsDepiction']=='1' or v['IsGroupOf']=='1' for v in pp):stats['native_group_or_depiction_excluded']+=1;continue
  title=metadata[k].get('Title','').lower();score=10*bool(re.search(r'river|sea\b|ocean|lake|open.water|snork|wad[ei]',title))
  rows.append({'image_id':k,'native_annotations':aa,'provenance':metadata[k],'source_split':'train','state':'UNREVIEWED_CONTEXT_CANDIDATE','automatic_swimmer_mapping':False,'review_priority':score})
 rows.sort(key=lambda r:(-r['review_priority'],r['image_id']))
 out=ROOT/'reports/openimages_water_activity_native.jsonl'
 out.write_text(''.join(json.dumps(r,ensure_ascii=True)+'\n' for r in rows),encoding='utf8')
 report={'caption_candidates':len(metadata),'native_training_boxes_scanned':n,'new_context_candidates':len(rows),'excluded':dict(stats),'automatic_approvals':0,'automatic_swimmer_mapping':False,'source':'Open Images original TRAIN metadata and all supplied native bounding boxes'}
 (ROOT/'reports/openimages_water_activity_discovery.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report),flush=True)
if __name__=='__main__':main()
