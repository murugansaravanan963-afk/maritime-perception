#!/usr/bin/env python3
"""Cross-group near-duplicate links for the pilot release.

Reads thumbnails from pilot_verify_images.py. Web-photo sources (DUTH, Open Images,
IALA10, AFO) re-publish the same photographs, so every cross-group pair there with
24x24 RGB correlation >= 0.85 is linked (conservative: over-linking only merges
split units). Other sources are fixed recordings; their high-correlation pairs are
open-water/horizon look-alikes. The strongest pair per group combination was viewed
on 2026-10-25 and none was the same capture, so they are recorded but not linked.
"""
import json,os
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
WORK=os.environ.get('PILOT_WORK','reports/pilot_r0')
WEB={'duth_swimmers_original','openimages_v7','iala10_original','afo_original_v1'}
WEB_LINK=0.85;OTHER_REPORT=0.93
# LaRS numbers consecutive clips of one session as separate sequences; viewed
# 2026-10-26: cross-sequence pairs >= 0.996 are the same scene. Link them.
SESSION_SOURCES={'lars_v1.0.0'};SESSION_LINK=0.95
# Viewed: identical photograph published in both DUTH and Open Images.
CONFIRMED=[('duth_swimmers:a52','openimages:8dec33d5efc66a73')]

def main():
 rs={}
 for l in (ROOT/'staging/annotation_review_pass.jsonl').read_text(encoding='utf8').splitlines():
  if l.strip():r=json.loads(l);rs[r['id']]=r
 z=np.load(ROOT/WORK/'thumbs.npz');ids=[str(x) for x in z['ids']]
 if set(ids)!=set(rs):raise ValueError('Thumbnails do not match the current annotation-pass manifest; rerun pilot_verify_images.py')
 t=z['thumbs'].astype(np.float32);t/=np.sqrt((t*t).sum(1,keepdims=True));sim=t@t.T
 group=[rs[i]['split_guard_group'] for i in ids];web=[rs[i]['source'] in WEB for i in ids]
 links=[];unlinked=0
 for i in range(len(ids)):
  for j in np.nonzero(sim[i,i+1:]>=min(WEB_LINK,OTHER_REPORT))[0]+i+1:
   if group[i]==group[j]:continue
   s=float(sim[i,j])
   if web[i] and web[j] and s>=WEB_LINK:links.append({'a':ids[i],'b':ids[j],'similarity':round(s,4),'basis':'web_photo_correlation'})
   elif rs[ids[i]]['source']==rs[ids[j]]['source'] in SESSION_SOURCES and s>=SESSION_LINK:links.append({'a':ids[i],'b':ids[j],'similarity':round(s,4),'basis':'same_session_clip_correlation'})
   elif s>=OTHER_REPORT:unlinked+=1
 have={(l['a'],l['b']) for l in links}|{(l['b'],l['a']) for l in links}
 for a,b in CONFIRMED:
  if (a,b) not in have:links.append({'a':a,'b':b,'similarity':None,'basis':'visually_confirmed_duplicate'})
 out={'thresholds':{'web_link':WEB_LINK,'other_report':OTHER_REPORT},'links':links,'non_web_pairs_not_linked':unlinked,
      'note':'Links merge split units; they do not remove images. Non-web look-alike pairs were spot-viewed (strongest per group pair) and were different captures.'}
 (ROOT/WORK/'near_duplicates.json').write_text(json.dumps(out,indent=1)+'\n',encoding='utf8')
 print(len(links),'links;',unlinked,'non-web look-alike pairs not linked')
if __name__=='__main__':main()
