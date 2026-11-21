#!/usr/bin/env python3
"""Produce full-frame and every-object evidence pages. Never creates approvals."""
import argparse,html,json,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps
from stage_candidates import ROOT,jsonl
from release_gate import NAMES,canonical_hash
COLORS=['#ff5b58','#c990ff','#3fe3d5','#ffd447','#ff66c9','#82ee62']
try: FONT=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',16)
except OSError: FONT=ImageFont.load_default()

def boxes(r):
 out=[(b['xyxy'],NAMES[b['class_id']],COLORS[b['class_id']]) for b in r['boxes']]
 out += [(b['native_xyxy'],'UNRESOLVED '+str(b.get('native_class',b.get('native_label',''))),'#ffffff') for b in r['unresolved_objects']]
 return out

def render(r,dest):
 im=Image.open(r['image']['path']).convert('RGB');w,h=im.size
 bw=1200;scale=min(1,bw/w);fh=round(h*scale);tiles=boxes(r);tw,th=240,190
 ch=50+fh+40+math.ceil(len(tiles)/5)*th
 canvas=Image.new('RGB',(bw,ch),'#151d2c');dr=ImageDraw.Draw(canvas)
 dr.text((10,8),r['id']+' | '+str(w)+'x'+str(h),font=FONT,fill='white')
 dr.text((10,28),'CANDIDATE / '+str(r['capture_group']),font=FONT,fill='#a9bacd')
 shown=im.resize((round(w*scale),fh));canvas.paste(shown,(0,50));dr=ImageDraw.Draw(canvas)
 for k,(xy,label,color) in enumerate(tiles,1):
  x1,y1,x2,y2=xy;v=[x1*scale,y1*scale+50,x2*scale,y2*scale+50]
  dr.rectangle(v,outline=color,width=2);dr.text((max(0,v[0]),max(50,v[1]-17)),str(k),font=FONT,fill=color,stroke_width=1,stroke_fill='black')
  pad=max(12,(x2-x1)*.2,(y2-y1)*.2)
  crop=im.crop((max(0,math.floor(x1-pad)),max(0,math.floor(y1-pad)),min(w,math.ceil(x2+pad)),min(h,math.ceil(y2+pad))))
  if crop.width and crop.height:
   crop=ImageOps.contain(crop,(tw-10,th-40),Image.Resampling.NEAREST if max(crop.size)<100 else Image.Resampling.LANCZOS);cx=(k-1)%5*tw;cy=50+fh+40+(k-1)//5*th
   canvas.paste(crop,(cx,cy+22));dr.text((cx,cy),str(k)+' '+label,font=FONT,fill=color)
 dr.text((10,50+fh+8),'Native boxes overlaid; detail crops below. Inspect for missing targets in full frame.',font=FONT,fill='white')
 canvas.save(dest,quality=94)
 return {'id':r['id'],'record_sha256':canonical_hash(r),'evidence_image':str(dest),'image_sha256':r['image']['sha256'],'box_count':len(r['boxes']),'unresolved_count':len(r['unresolved_objects'])}

def main():
 p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--priority');p.add_argument('--ids',type=Path);p.add_argument('--limit',type=int,default=0);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 ids=set(a.ids.read_text().splitlines()) if a.ids else None
 rs=[r for r in jsonl(a.manifest) if (not a.priority or r.get('review_priority')==a.priority) and (ids is None or r['id'] in ids)]
 if a.limit:rs=rs[:a.limit]
 a.out.mkdir(parents=True,exist_ok=True);evidence=[];sections=[]
 for i,r in enumerate(rs,1):
  dest=a.out/(f'{i:04d}.jpg');ev=render(r,dest);evidence.append(ev)
  sections.append('<section><h2>'+html.escape(r['id'])+'</h2><img loading="lazy" src="'+dest.name+'"><pre>'+html.escape(json.dumps({'issues':r['issues'],'checks_pending':r['checks_pending']},indent=2))+'</pre></section>')
 (a.out/'evidence.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in evidence),encoding='utf8')
 (a.out/'index.html').write_text('<!doctype html><meta charset="utf-8"><title>Maritime annotation review evidence</title><style>body{background:#111a26;color:#e5efff;font:16px system-ui;max-width:1220px;margin:auto}img{width:100%}section{margin:32px 0}pre{white-space:pre-wrap}</style><h1>Annotation evidence — unreleased candidates</h1><p>These pages contain no automatic approval controls. Record per-image findings bound to evidence and source hashes. A native box is not proof that all targets are labelled.</p>'+''.join(sections),encoding='utf8')
 print(json.dumps({'pages':len(evidence),'output':str(a.out)}))
if __name__=='__main__':main()
