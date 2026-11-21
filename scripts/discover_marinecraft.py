"""Select original TRAIN craft photos with all co-occurring annotations intact."""
import csv,json,re
from pathlib import Path
from collections import defaultdict,Counter
R=Path(__file__).resolve().parents[1]
names=dict(csv.reader((R/'research/openimages_boxable_classes.csv').open(encoding='utf-8-sig')))
limits={'Surfboard':1400,'Jet ski':400};target={k for k,v in names.items() if v in limits}
existing={Path(x['image']['path']).stem for p in (R/'staging/candidates').glob('openimages*.jsonl') for x in map(json.loads,p.read_text(encoding='utf8').splitlines())}
ids=set()
with (R/'raw/openimages/oidv6-train-annotations-bbox.csv').open(encoding='utf-8-sig',newline='') as f:
 for a in csv.DictReader(f):
  if a['LabelName'] in target and a['ImageID'] not in existing:ids.add(a['ImageID'])
print(json.dumps({'native_craft_images':len(ids)}),flush=True)
native=defaultdict(list)
with (R/'raw/openimages/oidv6-train-annotations-bbox.csv').open(encoding='utf-8-sig',newline='') as f:
 for a in csv.DictReader(f):
  if a['ImageID'] in ids:native[a['ImageID']].append(a)
prov={}
with (R/'raw/openimages/train-images-boxable-with-rotation.csv').open(encoding='utf-8-sig',newline='') as f:
 for a in csv.DictReader(f):
  if a['ImageID'] in ids:prov[a['ImageID']]=a
rows=[];ex=Counter()
for k,aa in native.items():
 if k not in prov:ex['missing_metadata']+=1;continue
 if any(a['IsGroupOf']=='1' or a['IsDepiction']=='1' for a in aa):ex['native_group_or_depiction']+=1;continue
 targets=[a for a in aa if a['LabelName'] in target]
 if len(targets)>4:ex['more_than_four_native_craft']+=1;continue
 if not prov[k].get('License') or not prov[k].get('AuthorProfileURL'):ex['missing_attribution']+=1;continue
 t=prov[k].get('Title','').lower();score=10*bool(re.search(r'sail|surf|ocean|sea\b|lake|river|water|canoe|kayak|harbo|coast',t))-10*bool(re.search(r'store|shop|statue|toy|drydock|repair|museum',t))
 rows.append({'image_id':k,'native_annotations':aa,'provenance':prov[k],'source_split':'train','review_priority':score,'state':'UNREVIEWED_CRAFT_AND_WATER_CONTEXT_CANDIDATE','selection_classes':sorted(set(names[a['LabelName']] for a in targets))})
rows.sort(key=lambda x:(-x['review_priority'],x['image_id']));selected={};counts=Counter()
for category,limit in limits.items():
 for r in rows:
  if counts[category]>=limit:break
  if category in r['selection_classes']:
   selected[r['image_id']]=r;counts[category]+=1
p=R/'reports/openimages_marinecraft_native.jsonl';assert not p.exists(),'Refuse overwrite discovery';p.write_text(''.join(json.dumps(a)+'\n' for a in selected.values()),encoding='utf8')
report={'native_target_images':len(ids),'eligible_context_review_candidates':len(rows),'selected':len(selected),'selection_counts':dict(counts),'excluded':dict(ex),'automatic_approvals':0,'generic_boat_and_person_labels_not_auto_mapped':True}
(R/'reports/openimages_marinecraft_discovery.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report),flush=True)
