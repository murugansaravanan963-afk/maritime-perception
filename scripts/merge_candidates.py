#!/usr/bin/env python3
"""Merge candidate inventories and real review evidence, never auto-admit labels."""
import argparse,json,hashlib,re,xml.etree.ElementTree as ET,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from stage_candidates import ROOT,jsonl
from release_gate import NAMES,canonical_hash
from audit_candidates import write_json,sha

def load_oostpoort_metadata():
 p=ROOT/'raw/oostpoort/Oostpoort Dataset.xlsx';ns={'s':'http://schemas.openxmlformats.org/spreadsheetml/2006/main'};sessions={}
 with zipfile.ZipFile(p) as z:
  strings=[''.join(si.itertext()) for si in ET.fromstring(z.read('xl/sharedStrings.xml')).findall('s:si',ns)]
  rows=ET.fromstring(z.read('xl/worksheets/sheet1.xml')).findall('s:sheetData/s:row',ns)
  for row in rows:
   cols={}
   for c in row.findall('s:c',ns):
    v=c.findtext('s:v','',ns)
    if c.get('t')=='s':v=strings[int(v)]
    elif c.get('t')=='inlineStr':v=''.join(c.find('s:is',ns).itertext())
    if v:cols[re.sub(r'\d+$','',c.get('r'))]=v
   exp=cols.get('B','')
   if re.fullmatch(r'exp\d+',exp) and re.fullmatch(r'\d{2}/\d{2}/\d{4}(?: - \d{2}/\d{2}/\d{4})?',cols.get('A','')):
    if exp in sessions:raise ValueError('Duplicate experiment metadata '+exp)
    sessions[exp]={'date_or_date_range_native':cols['A'],'start_time_native':cols.get('C'),'end_time_native':cols.get('D'),'site_window_native':cols.get('E'),'device':cols.get('K'),'resolution':cols.get('L'),'height_m':cols.get('J'),'angle_native':cols.get('I'),'native_positive_count':cols.get('O'),'native_box_count':cols.get('P')}
 return {'metadata_path':str(p),'metadata_sha256':sha(p.read_bytes()),'sessions':sessions,'split_guard':'Conservatively keep the whole fixed-camera Oostpoort site in one split, even though experiment dates differ.'}

def label_signature(r):
 return sorted((b['class_id'],*[round(float(x),5) for x in b['xyxy']]) for b in r['boxes'])

def exact_clusters(records):
 groups=defaultdict(list)
 for r in records:
  if r['image'].get('pixel_sha256'):groups[r['image']['pixel_sha256']].append(r)
 result=[]
 for ph,rs in groups.items():
  if len(rs)<2:continue
  sig=[label_signature(r) for r in rs];conflict=any(s!=sig[0] for s in sig[1:])
  cluster='pixel:'+ph
  for r in rs:
   r['duplicate_cluster']=cluster
   r['issues'].append('exact_duplicate_annotation_conflict' if conflict else 'exact_duplicate_needs_single_representative')
  result.append({'cluster':cluster,'ids':[r['id'] for r in rs],'annotation_conflict':conflict})
 return result

def summarize(rs):
 cc=Counter();bc=Counter();gp=defaultdict(set)
 for r in rs:
  if not r['image'].get('pixel_sha256'):continue
  bc.update(NAMES[b['class_id']] for b in r['boxes'])
  for c in {b['class_id'] for b in r['boxes']}:cc[NAMES[c]]+=1;gp[NAMES[c]].add(r['split_guard_group'])
 return {'images':len(rs),'mapped_positive_images':{n:cc[n] for n in NAMES},'mapped_boxes':{n:bc[n] for n in NAMES},'conservative_group_count':{n:len(gp[n]) for n in NAMES}}

def size_summary(rs):
 values={s:{n:[] for n in NAMES} for s in [640,960,1280]}
 for r in rs:
  im=r['image']
  if not im.get('width') or not im.get('pixel_sha256'):continue
  for b in r['boxes']:
   x1,y1,x2,y2=b['xyxy']
   if x2<=x1 or y2<=y1:continue
   for s in values:values[s][NAMES[b['class_id']]].append(min(x2-x1,y2-y1)*s/max(im['width'],im['height']))
 return {str(s):{n:{'boxes':len(v),'median_short_side_pixels':round(float(np.median(v)),2) if v else None,'short_side_below_4px':sum(x<4 for x in v),'short_side_below_8px':sum(x<8 for x in v)} for n,v in cs.items()} for s,cs in values.items()}

def corrected_candidates(records, patches):
 by={r['id']:r for r in records};seen=set()
 for p in patches:
  rid=p['id']
  if rid in seen or rid not in by:raise ValueError('Duplicate/unknown correction '+rid)
  seen.add(rid);base=by[rid];r=p['corrected_record']
  if canonical_hash(base)!=p['parent_record_sha256']:raise ValueError('Correction parent changed '+rid)
  for k in ['id','image','source','source_revision','source_annotation_sha256','source_split','split']:
   if r[k]!=base[k]:raise ValueError('Correction changed protected provenance '+rid)
  if r.get('parent_record_sha256')!=p['parent_record_sha256'] or not r.get('corrections'):raise ValueError('Missing explicit correction provenance')
  from audit_candidates import box_issues
  if any(type(b['class_id']) is not int or b['class_id'] not in range(6) for b in r['boxes']):raise ValueError('Invalid corrected class')
  problems=box_issues([{'native_class':b['class_id'],'xyxy':b['xyxy']} for b in r['boxes']],r['image']['width'],r['image']['height'])
  if problems:raise ValueError('Invalid corrected box '+str(problems))
  by[rid]=r
 return [by[r['id']] for r in records]

def merge():
 paths=sorted((ROOT/'staging/candidates').glob('*.jsonl'));rows=[];seen=set();source_files={};taxsha=sha((ROOT/'configs/taxonomy.json').read_bytes())
 for p in paths:
  source_files[p.name]=sha(p.read_bytes())
  for r in jsonl(p):
   if r['id'] in seen:raise ValueError('Duplicate candidate ID: '+r['id'])
   seen.add(r['id']);rows.append(r)
 patches=[]
 for cp in sorted((ROOT/'staging/corrections').glob('*.jsonl')):patches.extend(jsonl(cp))
 rows=corrected_candidates(rows,patches)
 for r in rows:r['candidate_record_sha256']=canonical_hash(r);r['split_guard_group']=r['capture_group']
 observations={}
 op=ROOT/'staging/visual_observations.jsonl'
 if op.exists():
  for o in jsonl(op):
   if o['id'] in observations:raise ValueError('Duplicate observation '+o['id'])
   observations[o['id']]=o
 metadata=load_oostpoort_metadata();write_json(ROOT/'reports/oostpoort_capture_metadata.json',metadata)
 linked={}
 # Conservative same-site pools based on full-frame evidence. These group
 # related events for split protection; they do NOT count as distinct sites.
 for tag,ids in [('river_recreation',[45,46,47,48,51,52,53,63,239,240,249,250,253]),('lake_recreation',[38,40,204,205]),('tropical_resort',[104,105,106,112,113,114,115,116])]:
  for i in ids:linked['proposal:lars:inhouse_seq_'+str(i)]='review_guard:lars:'+tag
 linked['proposal:lars:mastr_0811']=linked['proposal:lars:mastr_0812']='review_guard:lars:mastr_bathing_area_0811_0812'
 # Additional actual scene/session matches from lars_small_review_001.
 for i in [57,58,60,62,64,243,246,251,252,254]:linked['proposal:lars:inhouse_seq_'+str(i)]='review_guard:lars:river_recreation'
 for i in [4,17,102,421]:linked['proposal:lars:inhouse_seq_'+str(i)]='review_guard:lars:coastal_board_session'
 for i in [23,24,25,54,58,59,60,65,66]:linked['proposal:lars:davimar_seq_'+str(i)]='review_guard:lars:davimar_20210917_boat_trip'
 sds_group_path=ROOT/'reports/sds_capture_group_evidence.json'
 if sds_group_path.exists():
  sds_group_doc=json.loads(sds_group_path.read_text(encoding='utf-8'))
  if sds_group_doc['source_annotation_sha256']!=sha((ROOT/'raw/seadronessee_aux/instances_train_original.json').read_bytes()):raise ValueError('SDS grouping source changed')
  for g in sds_group_doc['groups']:linked[g['source_capture_group']]=g['split_guard_group']
 seaclips_guard_path=ROOT/'reports/seaclips_capture_group_evidence.json'
 if seaclips_guard_path.exists():
  gd=json.loads(seaclips_guard_path.read_text(encoding='utf8'))
  if gd['source_annotation_sha256']!=sha(Path(gd['source_annotation_path']).read_bytes()):raise ValueError('SeaClips grouping source changed')
  if gd['evidence_index_sha256']!=sha(Path(gd['evidence_index']).read_bytes()):raise ValueError('SeaClips grouping index changed')
  for ep,eh in gd['evidence_pages_sha256'].items():
   if sha(Path(ep).read_bytes())!=eh:raise ValueError('SeaClips group evidence changed')
  for g in gd['groups']:linked[g['source_capture_group']]=g['split_guard_group']
 afo_guard_path=ROOT/'reports/afo_capture_group_evidence.json'
 if afo_guard_path.exists():
  gd=json.loads(afo_guard_path.read_text(encoding='utf8'))
  if gd['source_manifest_sha256']!=source_files['afo.jsonl']:raise ValueError('AFO grouping source changed')
  for ep,eh in gd['evidence_files_sha256'].items():
   if sha(Path(ep).read_bytes())!=eh:raise ValueError('AFO grouping evidence changed')
  for g in gd['groups']:
   for original in g['source_capture_groups']:
    if original in linked and linked[original]!=g['split_guard_group']:raise ValueError('Conflicting AFO split guard')
    linked[original]=g['split_guard_group']
 quality_holds={};audit_hashes={}
 for qp in sorted((ROOT/'staging/quality_holds').glob('*.jsonl')):
  for q in jsonl(qp):
   if q['id'] in quality_holds or q['id'] not in seen:raise ValueError('Duplicate/unknown quality hold '+q['id'])
   ap=q['audit_path']
   if ap not in audit_hashes:audit_hashes[ap]=sha(Path(ap).read_bytes())
   if audit_hashes[ap]!=q['audit_sha256']:raise ValueError('Quality audit changed '+ap)
   quality_holds[q['id']]=q
 for r in rows:
  r['split_guard_group']=linked.get(r['capture_group'],r['capture_group'])
  if r['source']=='aboships':
   # Publisher dates are not proof of disjoint sites: protect the complete
   # moving-waterbus acquisition as one conservative route/campaign pool.
   r['split_guard_group']='campaign:aboships:finnish_archipelago_waterbus'
  if r['source']=='macvi_usv_dist':
   # Frame-number gaps split the archive into 7 pieces, but they may be one voyage
   # past the same harbour; keep the whole challenge set in one split.
   r['split_guard_group']='campaign:macvi:usv_dist_train'
  if r['source']=='waterscenes_original':
   # Protect the complete new acquisition until route/location metadata is
   # verified; timestamps/dates alone do not establish independent sites.
   r['split_guard_group']='campaign:waterscenes:original_surface_acquisition'
  if r['source']=='oostpoort':
   exp=re.search(r'Oos_(exp\d+)',r['id']).group(1)
   if exp not in metadata['sessions']:r['issues'].append('recording_metadata_missing')
   else:r['capture_metadata']={**metadata['sessions'][exp],'publisher_experiment':exp,'metadata_sha256':metadata['metadata_sha256']}
   r['split_guard_group']='site:oostpoort:fixed_camera'
  o=observations.get(r['id'])
  if o:
   if o['record_sha256']!=r['candidate_record_sha256'] or o['taxonomy_sha256']!=taxsha:raise ValueError('Stale visual observation '+r['id'])
   if o['image_sha256']!=r['image']['sha256']:raise ValueError('Observation image changed '+r['id'])
   if sha(Path(o['evidence_path']).read_bytes())!=o['evidence_sha256']:raise ValueError('Evidence image changed '+r['id'])
   r['visual_observation']=o
   if o['decision']=='quarantine':r['issues'].append('visual_semantic_or_completeness_failure')
  q=quality_holds.get(r['id'])
  if q:
   if q['image_sha256']!=r['image']['sha256']:raise ValueError('Quality hold image changed '+r['id'])
   r['quality_hold']=q;r['issues']+=q['issues']
  r['issues']=sorted(set(r['issues']))
 duplicates=exact_clusters(rows)
 for r in rows:
  o=r.get('visual_observation',{})
  if not r['image'].get('pixel_sha256'):state='IMAGE_UNAVAILABLE_OR_INVALID'
  elif r['issues']:state='QUARANTINE_OR_CONTEXT_REVIEW'
  elif o.get('decision')=='annotation_pass':state='ANNOTATION_REVIEWED_GROUPS_PENDING'
  else:state='FULL_FRAME_REVIEW_PENDING'
  r['review_state']=state
  # No canonical split, final review approval or training YAML is manufactured.
 bysource={s:summarize([r for r in rows if r['source']==s]) for s in sorted({r['source'] for r in rows})}
 decoded=[r for r in rows if r['image'].get('pixel_sha256')];structural=[r for r in decoded if not r['issues']];reviewed=[r for r in structural if r['review_state']=='ANNOTATION_REVIEWED_GROUPS_PENDING']
 report={'state':'MERGED_CANDIDATES_NOT_RELEASED','source_manifest_sha256':source_files,'candidate_records':len(rows),'decoded_images':len(decoded),'distinct_decoded_pixel_images':len({r['image']['pixel_sha256'] for r in decoded}),'missing_or_invalid_images':len(rows)-len(decoded),'states':dict(Counter(r['review_state'] for r in rows)),'by_source':bysource,'mapped_candidates':summarize(decoded),'native_geometry_and_mapping_unblocked_candidates':summarize(structural),'annotation_review_pass_pending_group_checks':summarize(reviewed),'duplicate_families':duplicates,'primary_surface_candidate_counts':summarize([r for r in decoded if r['viewpoint'] in ['boat','shore','surface']]),'aerial_auxiliary_candidate_counts':summarize([r for r in decoded if r['viewpoint']=='aerial']),'unresolved_viewpoint_candidate_counts':summarize([r for r in decoded if r['viewpoint']=='unknown']),'size_statistics_for_mapped_native_boxes_not_final_groundtruth':size_summary(decoded),'user_minimum_images_per_class':1000,'released_images':0,'training_started':False,'note':'Native mapped candidate counts include semantic ambiguity. annotation_pass is a visual observation, not completed split/dedup approval. Do not balance by copying adjacent frames or dropping missing classes.'}
 out=ROOT/'staging/candidate_master.jsonl';out.write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows),encoding='utf8')
 (ROOT/'staging/annotation_review_pass.jsonl').write_text(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in reviewed),encoding='utf8')
 write_json(ROOT/'reports/merged_candidate_status.json',report)
 lines=['# Fresh six-class dataset: actual merge status','','This is a review inventory, not a training release. Class counts are images containing a mapped box, not approved ground truth.','','| Class | All mapped candidates | Boat/shore candidates | Native geometry/mapping unblocked | Annotation check passed; groups pending |','|---|---:|---:|---:|---:|']
 for n in NAMES:lines.append('| '+n+' | '+str(report['mapped_candidates']['mapped_positive_images'][n])+' | '+str(report['primary_surface_candidate_counts']['mapped_positive_images'][n])+' | '+str(report['native_geometry_and_mapping_unblocked_candidates']['mapped_positive_images'][n])+' | '+str(report['annotation_review_pass_pending_group_checks']['mapped_positive_images'][n])+' |')
 lines+=['',f"Candidate images decoded: {len(decoded):,}; distinct decoded images: {report['distinct_decoded_pixel_images']:,}.",f"Exact duplicate families: {len(duplicates)}; conflicting label families: {sum(x['annotation_conflict'] for x in duplicates)}. All members stay quarantined until adjudicated.",f"Frames with saved visual observations: {len(observations)}. Released training images: 0.",'','## Remaining work','','Complete six-class full-frame labels, resolve duplicate conflicts and source terms, verify capture/near-duplicate groups, freeze disjoint validation/test groups, then pass release_gate.py. Keep primary swimmer coverage separate from aerial support. Existing run weights must not initialize the fresh benchmark.','','Oostpoort experiment dates are recovered from the original workbook. Its fixed site remains one conservative split group. LaRS same-scene clips seen during review have conservative cross-clip split guards. These are not claims that all source sessions are certified.']
 (ROOT/'reports/MERGE_STATUS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 print(json.dumps({k:report[k] for k in ['state','candidate_records','decoded_images','distinct_decoded_pixel_images','states']},indent=2))
 return report
if __name__=='__main__':merge()
