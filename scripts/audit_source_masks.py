"""Conservative quarantine hints for observed opaque source masks; never approves data."""
import json,sys,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy import ndimage
from PIL import Image
from stage_candidates import ROOT,jsonl
from release_gate import canonical_hash

def scan(r):
 p=Path(r['image']['path']);raw=p.read_bytes()
 if hashlib.sha256(raw).hexdigest()!=r['image']['sha256']:raise ValueError('Source changed '+r['id'])
 with Image.open(p) as im:
  w,h=im.size;im.thumbnail((640,640),Image.Resampling.NEAREST);a=np.asarray(im.convert('RGB'));mh,mw=a.shape[:2]
 mask=np.max(a,axis=2)<=3;lab,n=ndimage.label(mask);rects=[]
 for k,s in enumerate(ndimage.find_objects(lab),1):
  if s is None:continue
  yy,xx=s;bw=xx.stop-xx.start;bh=yy.stop-yy.start
  if bw<8 or bh<6 or bw*bh<80:continue
  occupancy=float(np.count_nonzero(lab[s]==k)/(bw*bh))
  if occupancy<.97:continue
  rects.append({'xyxy_proposal':[xx.start*w/mw,yy.start*h/mh,xx.stop*w/mw,yy.stop*h/mh],'sampled_black_fraction':occupancy,'sampled_size':[bw,bh]})
 return {'id':r['id'],'image_sha256':r['image']['sha256'],'image_pixel_sha256':r['image']['pixel_sha256'],'suspected_opaque_rectangles':rects}
def main():
 rows=list(jsonl(ROOT/'staging/candidates/seadronessee.jsonl'));out=ROOT/'reports/source_black_mask_audit.jsonl'
 with ThreadPoolExecutor(max_workers=4) as ex:results=list(ex.map(scan,rows))
 out.write_text(''.join(json.dumps(x)+'\n' for x in results),encoding='utf8')
 evidence_sha=hashlib.sha256(out.read_bytes()).hexdigest();native=json.loads((ROOT/'raw/seadronessee_aux/selected_train.json').read_text());cat4={a['image_id'] for a in native['annotations'] if a['category_id']==4};holds=[]
 for r,q in zip(rows,results):
  reasons=[]
  if q['suspected_opaque_rectangles']:reasons.append('suspected_opaque_source_mask_requires_resolution')
  if r['native_image_id'] in cat4:reasons.append('native_life_saving_category_requires_visual_adjudication')
  if reasons:holds.append({'id':r['id'],'image_sha256':r['image']['sha256'],'issues':reasons,'audit_path':str(out),'audit_sha256':evidence_sha,'reason':'Opaque masks observed in full-frame review. Native category4 cannot be blindly mapped: annotation23766/image4834 labels a clipped navigation buoy as life_saving_appliances. Whole-image hold, no automatic relabel or clipping.'})
 d=ROOT/'staging/quality_holds';d.mkdir(exist_ok=True);(d/'sds_source_integrity.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in holds),encoding='utf8')
 summary={'images_scanned':len(rows),'black_rectangle_hint_images':sum(bool(x['suspected_opaque_rectangles']) for x in results),'native_category4_scope_hold_images':len(cat4),'held_images_union':len(holds),'automatic_approvals':0,'method':'Exact-source-SHA checked; near-black rectangular component screen on <=640px nearest-neighbor preview; flag only, no claims that unflagged images are clean.'}
 (ROOT/'reports/source_black_mask_summary.json').write_text(json.dumps(summary,indent=2)+'\n',encoding='utf8');print(json.dumps(summary))
if __name__=='__main__':main()
