#!/usr/bin/env python3
"""Render review pages for unreviewed candidates with detector hints for MISSING targets.

A COCO detector (pinned original yolov8s) proposes boats, people and surfboards. Any
proposal not covered by a native box is drawn as a white numbered hint (P1, P2, ...).
Hints are only aids for spotting unlabelled objects; nothing is approved here.
Decisions are recorded separately with record_review.py, bound to these pages.

  python scripts/assist_review.py --sources lars_v1.0.0 --out reports/assist_lars_001
"""
import argparse,json,math
from collections import defaultdict
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from stage_candidates import ROOT,jsonl
from release_gate import NAMES,canonical_hash,digest
COLORS=['#ff5b58','#c990ff','#3fe3d5','#ffd447','#ff66c9','#82ee62']
COCO_HINTS={0:'person',8:'boat',37:'surfboard'}
try:FONT=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',15)
except OSError:FONT=ImageFont.load_default()
TW,TH=800,470

def unreviewed():
 """Base candidate records with no observation and no correction yet."""
 done={o['id'] for o in jsonl(ROOT/'staging/visual_observations.jsonl')}
 for p in (ROOT/'staging/corrections').glob('*.jsonl'):done|={c['id'] for c in jsonl(p)}
 out={}
 for p in sorted((ROOT/'staging/candidates').glob('*.jsonl')):
  for r in jsonl(p):
   if r['id'] not in done and not r['issues'] and not r['unresolved_objects'] and r['image'].get('pixel_sha256'):out[r['id']]=r
 return out

def covered(p,boxes):
 x1,y1,x2,y2=p;a=max(1e-6,(x2-x1)*(y2-y1))
 for b in boxes:
  u1,v1,u2,v2=b['xyxy'];iw=max(0,min(x2,u2)-max(x1,u1));ih=max(0,min(y2,v2)-max(y1,v1))
  if iw*ih/a>0.4:return True
 return False

def select(records,sources,per_group,stride,need=None):
 by=defaultdict(list)
 for r in records.values():
  if r['source'] in sources and (need is None or any(b['class_id']==need for b in r['boxes'])):by[r['capture_group']].append(r)
 picked=[]
 for g in sorted(by):
  rs=sorted(by[g],key=lambda r:r['id'])[::stride]
  picked+=rs[:per_group] if per_group else rs
 return picked

def tile(r,props,label):
 im=Image.open(r['image']['path']).convert('RGB');w,h=im.size;s=min(TW/w,(TH-20)/h)
 t=Image.new('RGB',(TW,TH),'#151d2c');t.paste(im.resize((round(w*s),round(h*s))),(0,20));d=ImageDraw.Draw(t)
 d.text((4,2),label,font=FONT,fill='white')
 def rect(xy,color,text,width):
  x1,y1,x2,y2=[v*s for v in xy];y1+=20;y2+=20;cx,cy=(x1+x2)/2,(y1+y2)/2
  # Keep tiny targets visible without changing their recorded geometry.
  x1,x2=min(x1,cx-5),max(x2,cx+5);y1,y2=min(y1,cy-5),max(y2,cy+5)
  d.rectangle([x1,y1,x2,y2],outline=color,width=width);d.text((x1,max(20,y1-15)),text,font=FONT,fill=color,stroke_width=1,stroke_fill='black')
 for i,b in enumerate(r['boxes'],1):rect(b['xyxy'],COLORS[b['class_id']],str(i),2)
 for p in props:rect(p['xyxy'],'white','P'+str(p['k'])+' '+p['coco'][:4],2)
 return t

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--sources',required=True);ap.add_argument('--out',type=Path,required=True)
 ap.add_argument('--per-group',type=int,default=0);ap.add_argument('--stride',type=int,default=1);ap.add_argument('--limit',type=int,default=0)
 ap.add_argument('--imgsz',type=int,default=1280);ap.add_argument('--ignore-bottom',type=float,default=1.0,help='drop hints whose bottom edge is below this fraction of image height (own camera platform)');ap.add_argument('--conf',type=float,default=0.25);ap.add_argument('--require-class',type=int,help='only frames with a native box of this class id');a=ap.parse_args()
 recs=select(unreviewed(),set(a.sources.split(',')),a.per_group,a.stride,a.require_class)
 if a.limit:recs=recs[:a.limit]
 if a.out.exists():raise FileExistsError('Use a new evidence directory')
 a.out=a.out.resolve();a.out.mkdir(parents=True)
 from ultralytics import YOLO
 model=YOLO(str(ROOT/'weights/yolov8s.pt'));index=[];page=[]
 def flush():
  n=len(index)//4+(1 if len(index)%4 else 0)-1
  sheet=Image.new('RGB',(2*TW,2*TH),'#0b111a')
  for j,t in enumerate(page):sheet.paste(t,((j%2)*TW,(j//2)*TH))
  dest=a.out/f'page_{n:04d}.jpg';sheet.save(dest,quality=92);ev=digest(dest.read_bytes())
  for row in index[-len(page):]:row['evidence_path']=str(dest);row['evidence_sha256']=ev
  page.clear()
 for start in range(0,len(recs),16):
  chunk=recs[start:start+16]
  res=model.predict([r['image']['path'] for r in chunk],imgsz=a.imgsz,conf=a.conf,classes=list(COCO_HINTS),verbose=False,device=0)
  for r,pr in zip(chunk,res):
   props=[]
   for xy,c,cf in zip(pr.boxes.xyxy.tolist(),pr.boxes.cls.tolist(),pr.boxes.conf.tolist()):
    if xy[3]>=a.ignore_bottom*r['image']['height']:continue
    if not covered(xy,r['boxes']):props.append({'k':len(props)+1,'xyxy':[round(v,2) for v in xy],'coco':COCO_HINTS[int(c)],'conf':round(cf,3)})
   slot='ABCD'[len(page)]
   page.append(tile(r,props,f"{slot} {r['id'][:60]}  view={r['viewpoint']}  boxes={len(r['boxes'])}  hints={len(props)}"))
   index.append({'id':r['id'],'slot':slot,'record_sha256':canonical_hash(r),'image_sha256':r['image']['sha256'],'boxes':len(r['boxes']),'proposals':props})
   if len(page)==4:flush()
 if page:flush()
 (a.out/'index.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in index),encoding='utf8')
 print(json.dumps({'records':len(index),'pages':math.ceil(len(index)/4),'with_hints':sum(bool(x['proposals']) for x in index),'out':str(a.out)}))
if __name__=='__main__':main()
