#!/usr/bin/env python3
"""Re-verify every annotation-passed image (bytes, decode, size, pixel hash) and
store small normalized thumbnails used for cross-group near-duplicate search."""
import hashlib,json,os,sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
def digest(b):return hashlib.sha256(b).hexdigest()
def check(r):
 im=r['image'];p=Path(im['path']);problems=[]
 data=p.read_bytes()
 if digest(data)!=im['sha256']:problems.append('image_hash_mismatch')
 with Image.open(p) as img:
  img.load();rgb=img.convert('RGB')
 if rgb.size!=(im['width'],im['height']):problems.append('image_dimension_mismatch')
 if digest(str(rgb.size).encode()+rgb.tobytes())!=im['pixel_sha256']:problems.append('decoded_pixel_hash_mismatch')
 t=np.asarray(rgb.resize((24,24),Image.BILINEAR),dtype=np.float32).reshape(-1)
 t=(t-t.mean())/(t.std()+1e-6)
 return r['id'],problems,t
def main():
 src=ROOT/'staging/annotation_review_pass.jsonl';out=ROOT/os.environ.get('PILOT_WORK','reports/pilot_r0');out.mkdir(parents=True,exist_ok=True)
 rs=[json.loads(l) for l in src.read_text(encoding='utf8').splitlines() if l.strip()]
 ids=[];thumbs=[];bad={}
 with ProcessPoolExecutor(8) as ex:
  for i,(rid,problems,t) in enumerate(ex.map(check,rs,chunksize=16)):
   ids.append(rid);thumbs.append(t)
   if problems:bad[rid]=problems
   if i%500==0:print(i,flush=True)
 np.savez_compressed(out/'thumbs.npz',ids=np.array(ids),thumbs=np.stack(thumbs).astype(np.float16))
 (out/'image_verification.json').write_text(json.dumps({'manifest':str(src),'manifest_sha256':digest(src.read_bytes()),'images_checked':len(ids),'failures':bad},indent=2)+'\n',encoding='utf8')
 print('checked',len(ids),'failures',len(bad))
if __name__=='__main__':main()
