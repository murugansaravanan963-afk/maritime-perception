#!/usr/bin/env python3
"""Append visual observations for an assist_review.py batch from a decisions file.

Decisions file lines: "<id> pass|hold [note]" (lines starting with # ignored).
Every id in the batch index must have exactly one decision. pass -> annotation_pass;
hold -> needs_review (the frame stays out of every release). Bound to the record,
taxonomy, image and evidence page hashes, like earlier observations.

  python scripts/record_review.py --batch reports/assist_lars_001
"""
import argparse,json
from datetime import datetime,timezone
from pathlib import Path
from stage_candidates import ROOT,jsonl
from release_gate import digest

def main():
 p=argparse.ArgumentParser();p.add_argument('--batch',type=Path,required=True);p.add_argument('--reviewer',default='Claude visual inspection');p.add_argument('--partial',action='store_true',help='batch was only partly viewed; record only frames with decisions, the rest stay unreviewed');a=p.parse_args()
 index=list(jsonl(a.batch/'index.jsonl'));decisions={}
 for line in (a.batch/'decisions.txt').read_text(encoding='utf8').splitlines():
  if not line.strip() or line.startswith('#'):continue
  # Tab-separated when ids contain spaces (e.g. "dsix:image (10)").
  rid,verdict,*note=line.split('\t',2) if '\t' in line else line.split(' ',2)
  if rid in decisions or verdict not in ('pass','hold'):raise ValueError('Duplicate or invalid decision: '+line)
  decisions[rid]=(verdict,note[0] if note else '')
 if a.partial:index=[x for x in index if x['id'] in decisions]
 ids={x['id'] for x in index}
 if set(decisions)!=ids:raise ValueError(f'Decisions do not cover the batch exactly: missing={sorted(ids-set(decisions))[:5]} extra={sorted(set(decisions)-ids)[:5]}')
 out=ROOT/'staging/visual_observations.jsonl';done={o['id'] for o in jsonl(out)}
 if ids&done:raise ValueError('Batch already recorded: '+str(sorted(ids&done)[:3]))
 tax=digest((ROOT/'configs/taxonomy.json').read_bytes());now=datetime.now(timezone.utc).isoformat();rows=[]
 for x in index:
  if digest(Path(x['evidence_path']).read_bytes())!=x['evidence_sha256']:raise ValueError('Evidence page changed '+x['evidence_path'])
  verdict,note=decisions[x['id']]
  findings=('Full frame checked for all six classes; native boxes match visible targets; detector hints for unlabelled boats/people/boards were each checked. '+note).strip() if verdict=='pass' else 'Held: '+(note or 'possible unlabelled target')+'. Whole frame excluded until corrected.'
  rows.append({'id':x['id'],'record_sha256':x['record_sha256'],'taxonomy_sha256':tax,'image_sha256':x['image_sha256'],'evidence_path':x['evidence_path'],'evidence_sha256':x['evidence_sha256'],
   'reviewed_by':a.reviewer,'reviewed_at':now,'inspection':f"Full frame on page {Path(x['evidence_path']).name} slot {x['slot']} with native boxes and COCO-detector hints for missing targets",
   'decision':'annotation_pass' if verdict=='pass' else 'needs_review','findings':findings,'release_approved':False,
   'remaining_checks':['source_terms','recording_group_verification','near_duplicate_review','independent_split_assignment']})
 with out.open('a',encoding='utf8') as f:f.write(''.join(json.dumps(r,ensure_ascii=False)+'\n' for r in rows))
 print(json.dumps({'recorded':len(rows),'pass':sum(r['decision']=='annotation_pass' for r in rows),'hold':sum(r['decision']!='annotation_pass' for r in rows)}))
if __name__=='__main__':main()
