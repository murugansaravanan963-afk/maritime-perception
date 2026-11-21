"""Discover original TRAIN waterbird photographs. Water context is not inferred.
All native labels are preserved for actual six-class full-frame review.
"""
import csv,json,re,hashlib
from pathlib import Path
from collections import defaultdict,Counter
R=Path(__file__).resolve().parents[1]
names=dict(csv.reader((R/'research/openimages_boxable_classes.csv').open(encoding='utf-8-sig')))
target={k for k,v in names.items() if v in {'Duck','Goose','Swan'}}
existing={Path(x['image']['path']).stem for p in (R/'staging/candidates').glob('openimages*.jsonl') for x in map(json.loads,p.read_text(encoding='utf8').splitlines())}
# First pass selects native species, not caption guesses. Second pass preserves
# every co-occurring native annotation including people, craft and generic birds.
ids=set()
with (R/'raw/openimages/oidv6-train-annotations-bbox.csv').open(encoding='utf-8-sig',newline='') as f:
 for a in csv.DictReader(f):
  if a['LabelName'] in target and a['ImageID'] not in existing:ids.add(a['ImageID'])
print(json.dumps({'native_species_images':len(ids)}),flush=True)
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
 species=[a for a in aa if a['LabelName'] in target]
 if len(species)>4:ex['more_than_four_species_proposals']+=1;continue
 if not prov[k].get('License') or not prov[k].get('AuthorProfileURL'):ex['missing_attribution']+=1;continue
 t=prov[k].get('Title','').lower();score=10*bool(re.search(r'swim|water|lake|pond|river|canal|harbo',t))-10*bool(re.search(r'flying|flight|zoo|statue|rubber|toy|roast|cook|stuffed',t))
 rows.append({'image_id':k,'native_annotations':aa,'provenance':prov[k],'source_split':'train','review_priority':score,'state':'UNREVIEWED_WATER_CONTEXT_CANDIDATE'})
rows.sort(key=lambda x:(-x['review_priority'],x['image_id']));selected=rows[:1600]
(R/'reports/openimages_waterbirds_native.jsonl').write_text(''.join(json.dumps(a)+'\n' for a in selected),encoding='utf8')
report={'species_images':len(ids),'eligible_context_review_candidates':len(rows),'selected':len(selected),'excluded':dict(ex),'automatic_approvals':0,'automatic_water_animal_mapping':False}
(R/'reports/openimages_waterbirds_discovery.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report),flush=True)
