#!/usr/bin/env python3
"""Refresh status from saved manifests; never invent approvals or splits."""
import json,hashlib
from pathlib import Path
from collections import Counter,defaultdict
from datetime import datetime,timezone
ROOT=Path(__file__).resolve().parents[1]
def rows(p):return [json.loads(x) for x in p.read_text(encoding='utf8').splitlines() if x.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def write(p,data):p.write_text(json.dumps(data,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def main():
 from release_gate import gate,NAMES
 now=datetime.now(timezone.utc).isoformat()
 master=rows(ROOT/'staging/candidate_master.jsonl');obs=rows(ROOT/'staging/visual_observations.jsonl')
 report=json.loads((ROOT/'reports/merged_candidate_status.json').read_text(encoding='utf8'))
 assert len(master)==report['candidate_records']
 assert len({o['id'] for o in obs})==len(obs)
 byid={r['id']:r for r in master};source=defaultdict(Counter)
 for o in obs:source[byid[o['id']]['source']][o['decision']]+=1
 corrected=[p for f in (ROOT/'staging/corrections').glob('*.jsonl') for p in rows(f)]
 assert len({p['id'] for p in corrected})==len(corrected)
 reviewed=[r for r in master if r['review_state']=='ANNOTATION_REVIEWED_GROUPS_PENDING']
 policy=json.loads((ROOT/'configs/release_policy.json').read_text())
 primary=[r for r in reviewed if r['viewpoint'] in policy['primary_viewpoints']]
 def count(rs):
  c=Counter()
  for r in rs:c.update({b['class_id'] for b in r['boxes']})
  return {NAMES[k]:c[k] for k in range(6)}
 total=count(reviewed);primary_count=count(primary)
 readiness={'updated_utc':now,'state':'NOT_READY_FOR_TRAINING','annotation_checked_images':len(reviewed),
  'annotation_checked_image_positives_all_views':total,'annotation_checked_image_positives_primary':primary_count,
  'minimum_training_images_per_class':policy['min_train_images_per_class'],
  'lower_bound_additional_annotation_checked_images_needed_per_class_before_holdouts':{n:max(0,policy['min_train_images_per_class']-total[n]) for n in NAMES},
  'note':'Counts are an upper bound before group/dedup/source checks and holdouts; not assigned training counts.',
  'canonical_training_images':0,'canonical_validation_images':0,'canonical_test_images':0,'balanced_release_ready':False}
 progress={'updated_utc':now,'state':'PARTIAL_VISUAL_REVIEW_NOT_DATASET_RELEASE','frames_inspected':len(obs),
  'explicit_corrected_images':len(corrected),'decisions':dict(Counter(o['decision'] for o in obs)),
  'by_source':{s:dict(c) for s,c in sorted(source.items())},'eligible_annotation_passes_pending_groups':len(reviewed),
  'raw_annotation_passes_excluded_by_other_inventory_holds':sum(o['decision']=='annotation_pass' for o in obs)-len(reviewed),
  'release_approvals':0,'ledger_sha256':sha(ROOT/'staging/visual_observations.jsonl')}
 write(ROOT/'reports/visual_review_progress.json',progress);write(ROOT/'reports/coverage_readiness.json',readiness)
 build={'updated_utc':now,'state':'MERGED_CANDIDATES_AND_PARTIAL_VISUAL_REVIEW; NOT_READY_FOR_TRAINING',
  'workspace':str(ROOT),'merged_candidate_images_decoded':report['decoded_images'],
  'distinct_decoded_images':report['distinct_decoded_pixel_images'],
  'mapped_candidate_images_per_class':report['mapped_candidates']['mapped_positive_images'],
  'primary_mapped_candidates_per_class':report['primary_surface_candidate_counts']['mapped_positive_images'],
  'detailed_visual_review':progress,'annotation_checked_coverage':readiness,
  'duplicate_conflict_families':sum(d['annotation_conflict'] for d in report['duplicate_families']),
  'reviewed_released_images':0,'new_training_started':False,'new_HEF_created':False,'current_policy':policy,
  'source_black_mask_audit':json.loads((ROOT/'reports/source_black_mask_summary_v2.json').read_text()),
  'main_remaining_work':['Complete six-class full-frame labels with >=1000 real training positives per class after holdouts','Finish source terms and original capture/site/photo-event verification','Adjudicate perceptual-neighbour hints and choose distinct real images','Freeze independent validation/test groups and pass release_gate.py'],
  'prepared_sol_launcher':'sol/train_fresh.sbatch','baidu_access':'unavailable; use other sources per user'}
 p=ROOT/'reports/test_results.json'
 if p.exists():build['software_validation']=json.loads(p.read_text())
 audits=sorted((ROOT/'reports').glob('reviewed_neighbours_*/summary.json'))
 if audits:
  p=audits[-1];build['reviewed_neighbour_audit']=json.loads(p.read_text())
  build['reviewed_neighbour_audit']['audit_report_path']=str(p)
  build['reviewed_neighbour_audit']['matches_current_reviewed_manifest']=build['reviewed_neighbour_audit']['manifest_sha256']==sha(ROOT/'staging/annotation_review_pass.jsonl')
 pairs=sorted((ROOT/'reports').glob('cross_guard_pairs_*/summary.json'))
 if pairs:
  p=pairs[-1];build['cross_guard_pair_review']=json.loads(p.read_text())
  build['cross_guard_pair_review']['report_path']=str(p)
  build['cross_guard_pair_review']['matches_current_reviewed_manifest']=build['cross_guard_pair_review'].get('audit_manifest_sha256')==sha(ROOT/'staging/annotation_review_pass.jsonl')
 write(ROOT/'reports/build_status.json',build)
 result=gate(master,[],policy,sha(ROOT/'configs/taxonomy.json'),verify_files=False)
 result['input_scope']='Staging candidates with deliberately unassigned splits and no final approvals. Not an export manifest. This staging gate does not reverify files.'
 result['updated_utc']=now;write(ROOT/'reports/release_gate.json',result)
 lines=['# Fresh Maritime6: current status','','**NOT READY FOR TRAINING. No training release or new Sol training has started.**','','Workspace: D:/Projects/maritime-dataset. The earlier project is preserved.','','## Saved evidence','',
  '- '+format(report['decoded_images'],',')+' decoded candidates; '+format(report['distinct_decoded_pixel_images'],',')+' distinct pixel images. Candidate counts are not reviewed training counts.',
  '- '+format(len(obs),',')+' saved full-frame observations: '+', '.join(str(v)+' '+k for k,v in sorted(progress['decisions'].items()))+'.',
  '- '+format(len(reviewed),',')+' annotation-passing images eligible for capture/source/duplicate checks.',
  '- '+str(len(corrected))+' accepted correction records, with source/image/taxonomy/evidence hashes.',
  '- '+str(build['duplicate_conflict_families'])+' conflicting exact duplicate families remain held.',
  '- Whole ABOships campaign, seven SeaClips pools, two SDS sites, fixed stream/Oostpoort sites and observed LaRS related scenes have protective guards; these do not certify independence.','',
  '## Annotation passed; final group/split approval pending','','| Class | All views | Primary surface | Minimum additional before holdouts |','|---|---:|---:|---:|']
 for n in NAMES:lines.append('| '+n+' | '+str(total[n])+' | '+str(primary_count[n])+' | '+str(readiness['lower_bound_additional_annotation_checked_images_needed_per_class_before_holdouts'][n])+' |')
 lines+=['','Image-positive counts overlap. They are not training/validation/test assignments. Additional images are needed beyond the last column to reserve holdouts and account for further exclusions.','',
  '## Release still requires','','At least 1,000 distinct real training positives per class, balanced classes, primary surface priority, complete labels and disjoint recording/site/photo-event holdouts. Aerial support never fills primary quotas. Copied/augmented frames do not add coverage.','',
  'The active policy retains curator coverage/diversity targets in configs/release_policy.json. No human sign-off prerequisite has been added.','',
  'The latest reviewed-image neighbour audit is saved in '+str(build.get('reviewed_neighbour_audit',{}).get('audit_report_path','not yet run'))+'. Cross-group pair review is saved in '+str(build.get('cross_guard_pair_review',{}).get('report_path','not yet run'))+'. Within-group redundancy review remains pending. Saved reports explicitly state whether they match the current reviewed manifest. Similarity is neither duplicate proof nor an independence certificate.','',
  'There is no training dataset.yaml release, fresh trained model, ONNX or HEF. 90% mAP@0.50 remains an evaluation target, not a guarantee.','',
  'Current evidence: reports/build_status.json, reports/coverage_readiness.json, reports/visual_review_progress.json and reports/merged_candidate_status.json.']
 (ROOT/'STATUS.md').write_text('\n'.join(lines)+'\n',encoding='utf8')
 hashes={}
 for folder in ['configs','scripts','tests','sol']:
  for p in sorted((ROOT/folder).rglob('*')):
   if p.is_file() and '__pycache__' not in p.parts:hashes[str(p.relative_to(ROOT))]=sha(p)
 for name in ['README.md','STATUS.md','requirements-audit.txt']:
  p=ROOT/name
  if p.exists():hashes[name]=sha(p)
 write(ROOT/'reports/workspace_file_hashes.json',hashes)
 print(json.dumps({'observations':len(obs),'annotation_eligible':len(reviewed),'class_counts':total,'released':0},indent=2))
if __name__=='__main__':main()
