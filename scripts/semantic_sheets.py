#!/usr/bin/env python3
"""Full-frame source-box sheets for actual semantic review. Proposals only."""
import argparse,csv,json,re,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps
from stage_candidates import ROOT,jsonl
from release_gate import canonical_hash
CLASSES=dict(csv.reader((ROOT/'research/openimages_boxable_classes.csv').open(encoding='utf-8-sig')))
HUMAN={'Person','Man','Woman','Boy','Girl'}
CRAFT={'Boat','Watercraft','Canoe','Surfboard','Jet ski','Sailboat'}
ANIMALS={'Animal','Bird','Duck','Goose','Swan','Dog','Cat','Marine mammal','Sea turtle','Turtle','Dolphin','Whale','Horse'}
FONT=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',17)

def iou(a,b):
 x=max(0,min(a[2],b[2])-max(a[0],b[0]));y=max(0,min(a[3],b[3])-max(a[1],b[1]));inter=x*y
 return inter/((a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter) if inter else 0

def proposals(r):
 out=[];w,h=r['image']['width'],r['image']['height']
 for ix,a in enumerate(r['native_annotations']):
  n=CLASSES.get(a['LabelName'],a['LabelName']);kind='p' if n in HUMAN else 'v' if n in CRAFT else 'a' if n in ANIMALS else None
  if not kind:continue
  xy=[float(a[k])*(w if k.startswith('X') else h) for k in ['XMin','YMin','XMax','YMax']]
  out.append({'kind':kind,'native_name':n,'native_rows':[ix],'xyxy':xy})
 result=[]
 for q in sorted(out,key=lambda q:((q['xyxy'][2]-q['xyxy'][0])*(q['xyxy'][3]-q['xyxy'][1])),reverse=True):
  match=next((b for b in result if b['kind']==q['kind'] and iou(b['xyxy'],q['xyxy'])>.85),None)
  if match:match['native_rows']+=q['native_rows']
  else:result.append(q)
 counters={'p':0,'v':0,'a':0}
 for q in result:q['key']=q['kind']+str(counters[q['kind']]);counters[q['kind']]+=1
 return result

def draw(r,qs):
 im=Image.open(r['image']['path']).convert('RGB');dr=ImageDraw.Draw(im);colors={'p':'#ff55c6','v':'#31e8db','a':'#91ed55'}
 for q in qs:
  x1,y1,x2,y2=q['xyxy'];c=colors[q['kind']];dr.rectangle([x1,y1,x2,y2],outline=c,width=2);dr.text((x1,max(0,y1-19)),q['key'],font=FONT,fill=c,stroke_width=1,stroke_fill='black')
 return im

def main():
 p=argparse.ArgumentParser();p.add_argument('--start',type=int,default=0);p.add_argument('--count',type=int,default=128);p.add_argument('--source',default='openimages_people');p.add_argument('--target-kind',choices=['p','a','v'],default='p');p.add_argument('--prefix',default='people_semantic');p.add_argument('--require-unobserved',action='store_true',help='Exclude every image id with an existing saved visual observation');a=p.parse_args()
 rows=list(jsonl(ROOT/f'staging/candidates/{a.source}.jsonl'));scored=[]
 observed={r['id'] for r in jsonl(ROOT/'staging/visual_observations.jsonl')} if a.require_unobserved else set()
 for r in rows:
  if r['id'] in observed:continue
  ps=proposals(r);people=[q for q in ps if q['kind']==a.target_kind];title=r.get('source_attribution',{}).get('Title','').lower()
  score=10*bool(re.search(r'swim|sea\b|ocean|beach|snork|bathing|river|lake|water|surf',title))-10*bool(re.search(r'pool|triathl|ironman|marathon|fashion|portrait|festival',title))+3*(len(people)<=3)-4*(len(people)>8)
  if not people:continue
  scored.append((score,r['id'],r,ps))
 scored.sort(key=lambda x:(-x[0],x[1]));selected=scored[a.start:a.start+a.count];out=ROOT/'reports'/a.prefix
 if out.exists():raise FileExistsError('Evidence directory already exists; choose a new --prefix: '+str(out))
 out.mkdir(parents=True)
 index=[]
 for offset,(_,rid,r,ps) in enumerate(selected):index.append({'ordinal':a.start+offset,'id':rid,'candidate_record_sha256':canonical_hash(r),'proposals':ps,'image':r['image']})
 (out/f'index_{a.start:04d}.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in index),encoding='utf8')
 # Four full frames per page. Original images stay available for closer inspection.
 for base in range(0,len(selected),4):
  page=Image.new('RGB',(1600,1400),'#172231');d=ImageDraw.Draw(page)
  for j,(_,rid,r,ps) in enumerate(selected[base:base+4]):
   x=j%2*800;y=j//2*700;im=ImageOps.contain(draw(r,ps),(795,590));page.paste(im,(x,y+45))
   d.text((x+4,y+4),str(a.start+base+j)+' '+rid,font=FONT,fill='white')
   d.text((x+4,y+615),' | '.join(q['key']+':'+q['native_name'] for q in ps)[:90],font=FONT,fill='white')
   d.text((x+4,y+645),r.get('source_attribution',{}).get('Title','')[:75],font=FONT,fill='#a9bdcf')
  page.save(out/f'page_{(a.start+base)//4:04d}.jpg',quality=95)
 print(json.dumps({'output':str(out),'frames':len(selected),'pages':math.ceil(len(selected)/4),'start':a.start}))
if __name__=='__main__':main()
