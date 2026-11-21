#!/usr/bin/env python3
"""Find original Open Images TRAIN people candidates; garment/cap != swimmer."""
import csv,json
from collections import defaultdict,Counter
from pathlib import Path
from audit_candidates import ROOT,write_json,sha

def main():
 with (ROOT/'research/openimages_boxable_classes.csv').open(encoding='utf-8-sig',newline='') as f:names=dict(csv.reader(f))
 targets={c for c,n in names.items() if n in ['Swimwear','Swim cap']};people={c for c,n in names.items() if n in ['Person','Man','Woman','Boy','Girl']};pool={c for c,n in names.items() if n=='Swimming pool'}
 path=ROOT/'raw/openimages/oidv6-train-annotations-bbox.csv';candidates=set();positive=Counter();n=0
 with path.open(encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f):
   n+=1
   if row['LabelName'] in targets and row['IsDepiction']=='0' and row['IsGroupOf']=='0':candidates.add(row['ImageID']);positive[names[row['LabelName']]]+=1
   if n%3000000==0:print('People source search',n,'native training boxes scanned',flush=True)
 native=defaultdict(list)
 with path.open(encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f):
   if row['ImageID'] in candidates:native[row['ImageID']].append(row)
 selected=[];stats=Counter()
 for rid,aa in native.items():
  pp=[a for a in aa if a['LabelName'] in people]
  stats['images_with_explicit_people_boxes']+=bool(pp)
  if not pp:continue
  if any(a['LabelName'] in pool for a in aa):stats['native_swimming_pool_images_excluded']+=1;continue
  if any(a['LabelName'] in people|targets and (a['IsDepiction']=='1' or a['IsGroupOf']=='1') for a in aa):stats['depiction_or_group_people_images_excluded']+=1;continue
  selected.append({'id':'openimages_people:'+rid,'image_id':rid,'source':'openimages_v7','source_split':'train','native_annotations':aa,'state':'UNREVIEWED_CONTEXT_CANDIDATE','automatic_swimmer_mapping':False})
 metadata={}
 with (ROOT/'raw/openimages/train-images-boxable-with-rotation.csv').open(encoding='utf-8-sig',newline='') as f:
  keep={x['image_id'] for x in selected}
  for row in csv.DictReader(f):
   if row['ImageID'] in keep:metadata[row['ImageID']]=row
 out=ROOT/'reports/openimages_people_native_candidates.jsonl'
 with out.open('w',encoding='utf8') as f:
  for r in selected:
   r['provenance']=metadata.get(r['image_id']);r['original_cvdf_url']='https://open-images-dataset.s3.amazonaws.com/train/'+r['image_id']+'.jpg';r['native_annotation_sha256']=sha(json.dumps(r['native_annotations'],sort_keys=True).encode());f.write(json.dumps(r,ensure_ascii=False)+'\n')
 report={'state':'ORIGINAL_SOURCE_DISCOVERY_NOT_SWIMMER_GROUNDTRUTH','train_boxes_scanned':n,'native_garment_cap_box_counts':dict(positive),'native_target_photo_image_ids':len(candidates),'context_candidates_with_people_boxes_no_native_pool_box':len(selected),'metadata_found':len(metadata),'filter_statistics':dict(stats),'images_downloaded':0,'confirmed_on_water_people_images':0,'admitted_images':0,'caveat':'Swimwear/cap is NOT a person_in_water label. Missing pool box does not prove open water. Full image context, box extent and all co-occurring target classes require review. No Baidu or publisher tests used.'}
 write_json(ROOT/'reports/openimages_people_discovery.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
