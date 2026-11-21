#!/usr/bin/env python3
"""Bounded perceptual-neighbour triage across proposed split-guard groups.
Low dHash distance is a review hint, especially unreliable for fixed water backgrounds.
It never certifies uniqueness or changes labels, approvals or split assignments.
"""
import json
from collections import Counter
from pathlib import Path
import numpy as np
from audit_candidates import ROOT,write_json,sha

def main():
 p=ROOT/'staging/candidate_master.jsonl';rs=[json.loads(x) for x in p.read_text(encoding='utf8').splitlines()];rs=[r for r in rs if 'dhash64' in r['image']]
 vals=np.array([int(r['image']['dhash64'],16) for r in rs],dtype=np.uint64);lut=np.array([i.bit_count() for i in range(256)],dtype=np.uint8);pairs={};total=0;sourcepairs=Counter();guard=np.array([r['split_guard_group'] for r in rs],dtype=object)
 for i,v in enumerate(vals):
  d=lut[np.bitwise_xor(v,vals).view(np.uint8).reshape(-1,8)].sum(axis=1)
  eligible=np.flatnonzero((d<=4)&(guard!=guard[i]));total+=sum(j>i for j in eligible)
  # Keep at most three closest neighbours per record, declared explicitly.
  eligible=sorted(map(int,eligible),key=lambda j:(int(d[j]),rs[j]['id']))[:3]
  for j in eligible:
   if rs[j]['image']['pixel_sha256']==rs[i]['image']['pixel_sha256']:continue
   a,b=sorted([i,j]);pairs[(a,b)]={'a':rs[a]['id'],'b':rs[b]['id'],'dhash_distance':int(d[j]),'guard_a':rs[a]['split_guard_group'],'guard_b':rs[b]['split_guard_group'],'sources':[rs[a]['source'],rs[b]['source']]}
  if (i+1)%3000==0:print('Perceptual triage',i+1,'/',len(rs),flush=True)
 out=ROOT/'reports/merged_neighbour_triage.jsonl';out.write_text(''.join(json.dumps(v)+'\n' for k,v in sorted(pairs.items())),encoding='utf8')
 for v in pairs.values():sourcepairs[' + '.join(sorted(v['sources']))]+=1
 report={'state':'REVIEW_HINTS_NOT_DUPLICATE_CERTIFICATION','candidate_manifest_sha256':sha(p.read_bytes()),'images_compared':len(rs),'threshold':'64-bit dHash Hamming <= 4','candidate_pairs_across_different_guard_groups_before_cap':int(total),'reported_pairs':len(pairs),'cap':'At most three closest different-guard neighbours per image; undirected union. Not an exhaustive near-duplicate list.','source_pair_counts':dict(sourcepairs),'rejections_or_merges_applied':0,'warning':'Fixed backgrounds can dominate dHash despite distinct objects/frames. Verify image content, timestamps and source recordings. Absence of a match does not prove distinct capture events.'}
 write_json(ROOT/'reports/merged_neighbour_triage.json',report);print(json.dumps(report,indent=2))
if __name__=='__main__':main()
