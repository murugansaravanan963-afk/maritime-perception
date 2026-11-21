"""Render complete RGB frames and all LOCAL detector proposals; no approvals."""
from pathlib import Path
import argparse,json,math
from PIL import Image,ImageOps,ImageDraw,ImageFont
from stage_candidates import ROOT,jsonl
from release_gate import canonical_hash
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--count',type=int,default=400);ap.add_argument('--out',required=True);a=ap.parse_args()
 source=ROOT/'reports/yangtze983_proposals_001'
 if not (source/'summary.json').exists():raise ValueError('Proposal generation not complete')
 rs={r['id']:r for r in jsonl(ROOT/'staging/candidates/yangtze983.jsonl')}
 observed={r['id'] for r in jsonl(ROOT/'staging/visual_observations.jsonl')}
 items=[]
 for p in jsonl(source/'predictions.jsonl'):
  r=rs[p['id']]
  if p['candidate_record_sha256']!=canonical_hash(r) or p['image_sha256']!=r['image']['sha256']:raise ValueError('Stale proposal')
  ps=p['proposals'];area=r['image']['width']*r['image']['height']
  if r['id'] in observed or not 1<=len(ps)<=3:continue
  if max((q['xyxy'][2]-q['xyxy'][0])*(q['xyxy'][3]-q['xyxy'][1]) for q in ps)<.025*area:continue
  items.append((r,p))
 items.sort(key=lambda pair:canonical_hash({'id':pair[0]['id'],'seed':20261025}))
 items=items[:a.count];out=ROOT/'reports'/a.out
 if out.exists():raise FileExistsError(out)
 out.mkdir();font=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',17);index=[]
 for base in range(0,len(items),4):
  canvas=Image.new('RGB',(1600,1500),'#172432');d=ImageDraw.Draw(canvas)
  for j,(r,p) in enumerate(items[base:base+4]):
   n=base+j;xx=j%2*800;yy=j//2*750
   im=Image.open(r['image']['path']).convert('RGB');draw=ImageDraw.Draw(im)
   for q in p['proposals']:
    b=q['xyxy'];draw.rectangle(b,outline='#40ded0',width=3);draw.text((b[0],max(0,b[1]-22)),q['proposal_id'],font=font,fill='#40ded0',stroke_width=1,stroke_fill='black')
   shown=ImageOps.contain(im,(795,520));canvas.paste(shown,(xx,yy+35))
   d.text((xx+4,yy+4),str(n)+' '+r['id'],font=font,fill='white')
   tilew=min(390,790//len(p['proposals']))
   for bi,q in enumerate(p['proposals']):
    xy=q['xyxy'];pad=max(8,(xy[2]-xy[0])*.08,(xy[3]-xy[1])*.08)
    crop=im.crop((max(0,math.floor(xy[0]-pad)),max(0,math.floor(xy[1]-pad)),min(im.width,math.ceil(xy[2]+pad)),min(im.height,math.ceil(xy[3]+pad))))
    crop=ImageOps.contain(crop,(tilew-4,178));canvas.paste(crop,(xx+bi*tilew,yy+563))
    d.text((xx+bi*tilew,yy+543),q['proposal_id']+' boat PROPOSAL',font=font,fill='#40ded0')
   index.append({'ordinal':n,'id':r['id'],'candidate_record_sha256':canonical_hash(r),'image':r['image'],'proposals':p['proposals'],'weight_sha256':p['weight_sha256'],'display_limit':[795,520],'page':str(out/f'page_{base//4:04d}.jpg')})
  canvas.save(out/f'page_{base//4:04d}.jpg',quality=96)
 (out/'index.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in index),encoding='utf8')
 print(json.dumps({'images':len(items),'pages':math.ceil(len(items)/4),'status':'UNOBSERVED_PROPOSALS_NO_GROUND_TRUTH','out':str(out)}))
if __name__=='__main__':main()
