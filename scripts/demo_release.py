#!/usr/bin/env python3
"""Class-demo release: random 80/10/10 split of the reviewed images, near-duplicates kept together.

Unlike the pilot releases (whole recordings per split), this is the usual image-level split:
each recording is cut into short runs of consecutive frames (natural id order), runs that
share a near-identical frame (thumbnail correlation >= DUP, practically identical) are merged, and whole runs are
assigned at random. Neighbouring video frames therefore never sit on both sides, but other
parts of the same recording can. All viewpoints are evaluated. Report it as:
"random image-level split; consecutive frames and near-duplicates grouped".

  python scripts/demo_release.py --out reports/demo_d1/split_audit.json --publish releases/maritime6_demo_d1
"""
import argparse,json,os,random,re
from collections import Counter,defaultdict
from pathlib import Path
import numpy as np
from pilot_release import ROOT,WORK,assign,check,publish,units,read_jsonl,digest
from release_gate import NAMES
RUN=12;DUP=0.99;FRACTIONS={'train':0.8,'val':0.1,'test':0.1}

def natural(s):return [int(t) if t.isdigit() else t for t in re.split(r'(\d+)',s)]

def runs(records,links,policy):
 """Pilot unit (recording + cross-source duplicate links) cut into runs of consecutive frames.

 d1 also merged runs sharing a practically identical frame; through a video that chains
 run to run (blocks of 100-588 frames), which defeated the run design. d2 drops that
 merge; fixed cameras (near-identical all day) stay whole instead."""
 unit=units(records,links);by=defaultdict(list);whole=set(policy.get('whole_unit_groups',[]))
 for r in records:by[unit[r['id']]].append(r)
 run_of={}
 for u,rs in by.items():
  keep=any(r['split_guard_group'] in whole for r in rs)
  for k,r in enumerate(sorted(rs,key=lambda r:natural(r['id']))):run_of[r['id']]=u if keep else f'{u}:{k//RUN}'
 if not policy.get('merge_identical_runs',False):return {rid:'run:'+digest(r.encode())[:16] for rid,r in run_of.items()}
 z=np.load(ROOT/WORK/'thumbs.npz');ids=[str(x) for x in z['ids']];t=z['thumbs'].astype(np.float32);t/=np.sqrt((t*t).sum(1,keepdims=True))
 parent={}
 def find(x):
  parent.setdefault(x,x)
  while parent[x]!=x:parent[x]=parent[parent[x]];x=parent[x]
  return x
 sim=t@t.T
 for i,j in zip(*np.nonzero(np.triu(sim>=DUP,1))):parent[find(run_of[ids[i]])]=find(run_of[ids[j]])
 return {rid:'run:'+digest(find(r).encode())[:16] for rid,r in run_of.items()}

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);ap.add_argument('--publish',type=Path);a=ap.parse_args()
 pb=(ROOT/os.environ.get('DEMO_POLICY','configs/demo_release_policy.json')).read_bytes();policy=json.loads(pb)
 records=read_jsonl(ROOT/'staging/annotation_review_pass.jsonl')
 ver=json.loads((ROOT/WORK/'image_verification.json').read_text(encoding='utf8'))
 if ver['manifest_sha256']!=digest((ROOT/'staging/annotation_review_pass.jsonl').read_bytes()) or ver['failures']:raise ValueError('Image verification stale or failed')
 links=json.loads((ROOT/WORK/'near_duplicates.json').read_text(encoding='utf8'))['links']
 run_of=runs(records,links,policy);members=defaultdict(list)
 for r in records:members[run_of[r['id']]].append(r)
 if policy.get('assignment')=='balanced':
  # Seeded annealing over whole runs toward each class's val/test share (and per-source share
  # when source_balance is set), so one large run cannot make up a class's whole val set.
  split=assign(records,policy,run_of)
 else:
  # Seeded random assignment of whole runs; the seed is fixed in the policy, not chosen by results.
  order=sorted(members);rng=random.Random(policy['seed']);rng.shuffle(order);total=len(records);split={};filled=Counter()
  for u in order:
   s=min(FRACTIONS,key=lambda k:filled[k]/total-FRACTIONS[k]);split[u]=s;filled[s]+=len(members[u])
 out=[]
 for r in records:
  r=dict(r);r['pilot_unit']=run_of[r['id']];r['split']=split[r['pilot_unit']];out.append(r)
 result=check(out,policy);result['dropped_train_only_records_in_holdout_units']=0
 # Run statistics go to the report only; the release's audit.json must equal check() for preflight.
 report={**result,'runs':len(members),'largest_runs':sorted((len(v) for v in members.values()),reverse=True)[:5]}
 a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
 print(json.dumps({k:report[k] for k in ['passed','images_per_split','images_containing_class','runs','largest_runs']},indent=1));print('errors:',result['errors'][:10])
 if a.publish:publish(out,policy,pb,result,a.publish)
 raise SystemExit(0 if result['passed'] else 2)
if __name__=='__main__':main()
