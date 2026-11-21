"""Versioned, conservative source-mask screen; no semantic approvals."""
import json,hashlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
from scipy import ndimage
from PIL import Image
from stage_candidates import ROOT,jsonl

def scan(r):
 p=Path(r['image']['path']);data=p.read_bytes()
 if hashlib.sha256(data).hexdigest()!=r['image']['sha256']:raise ValueError('Source image changed '+r['id'])
 with Image.open(p) as im:
  w,h=im.size;im.thumbnail((1280,1280),Image.Resampling.NEAREST);a=np.asarray(im.convert('RGB'));mh,mw=a.shape[:2]
 mask=np.max(a,axis=2)<=3;lab,n=ndimage.label(mask);rects=[]
 for k,s in enumerate(ndimage.find_objects(lab),1):
  if s is None:continue
  yy,xx=s;bw=xx.stop-xx.start;bh=yy.stop-yy.start
  if min(bw,bh)<4 or bw*bh<16:continue
  occupancy=float(np.count_nonzero(lab[s]==k)/(bw*bh))
  if occupancy<.98:continue
  rects.append({'xyxy_proposal':[xx.start*w/mw,yy.start*h/mh,xx.stop*w/mw,yy.stop*h/mh],'sampled_black_fraction':occupancy,'sampled_size':[bw,bh]})
 return {'id':r['id'],'image_sha256':r['image']['sha256'],'suspected_opaque_rectangles_v2':rects}

def main():
 rows=list(jsonl(ROOT/'staging/candidates/seadronessee.jsonl'));old={q['id']:q for q in jsonl(ROOT/'reports/source_black_mask_audit.jsonl')};obs={o['id']:o for o in jsonl(ROOT/'staging/visual_observations.jsonl')}
 out=ROOT/'reports/source_black_mask_audit_v2.jsonl'
 if out.exists():raise FileExistsError('Preserve immutable versioned audit '+str(out))
 with ThreadPoolExecutor(max_workers=4) as ex:
  results=[]
  for n,q in enumerate(ex.map(scan,rows),1):
   prior=old[q['id']]
   if prior['image_sha256']!=q['image_sha256']:raise ValueError('Source image differs from v1 audit')
   q['v1_rectangles_retained']=prior['suspected_opaque_rectangles']
   o=obs.get(q['id'],{})
   if o.get('decision')=='quarantine' and ('opaque' in o.get('findings','').lower() or 'black source mask' in o.get('findings','').lower()):q['visual_mask_observation']={k:o[k] for k in ['evidence_path','evidence_sha256','findings']}
   results.append(q)
   if n%300==0:print('Mask v2 checked',n,'/',len(rows),flush=True)
 out.write_text(''.join(json.dumps(q)+'\n' for q in results),encoding='utf-8');audit_sha=hashlib.sha256(out.read_bytes()).hexdigest()
 native=json.loads((ROOT/'raw/seadronessee_aux/selected_train.json').read_text());cat4={a['image_id'] for a in native['annotations'] if a['category_id']==4};holds=[]
 for r,q in zip(rows,results):
  reasons=[]
  if q['suspected_opaque_rectangles_v2'] or q['v1_rectangles_retained'] or q.get('visual_mask_observation'):reasons.append('suspected_opaque_source_mask_requires_resolution')
  if r['native_image_id'] in cat4:reasons.append('native_life_saving_category_requires_visual_adjudication')
  if reasons:holds.append({'id':r['id'],'image_sha256':r['image']['sha256'],'issues':reasons,'audit_path':str(out),'audit_sha256':audit_sha,'reason':'Version2 screens smaller rectangles at 1280px; retains all prior mask hints and explicit visual mask observations. Category4 remains held. Screening never approves an unflagged image.'})
 (ROOT/'staging/quality_holds/sds_source_integrity.jsonl').write_text(''.join(json.dumps(q)+'\n' for q in holds),encoding='utf-8')
 summary={'images_scanned':len(rows),'v2_black_rectangle_hint_images':sum(bool(q['suspected_opaque_rectangles_v2']) for q in results),'all_versions_or_visual_mask_hint_images':sum(bool(q['suspected_opaque_rectangles_v2'] or q['v1_rectangles_retained'] or q.get('visual_mask_observation')) for q in results),'native_category4_scope_hold_images':len(cat4),'held_images_union':len(holds),'automatic_approvals':0,'method':'Exact image SHA checked. <=1280px nearest-neighbor preview, near-black connected rectangular components >=4x4px and >=98% occupancy. All v1 hints retained; explicit visual masks retained. An unflagged image is never declared clean.'}
 (ROOT/'reports/source_black_mask_summary_v2.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf-8');print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
