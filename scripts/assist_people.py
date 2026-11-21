#!/usr/bin/env python3
"""Review pages for Open Images photos quarantined as 'person_in_water_status_not_established'.

Each person-type native box (Person, Man, Woman, Boy, Girl, Human body) is drawn with a
number. Other native objects that could be targets (boats, boards, animals...) are drawn
in white with their name, plus COCO-detector hints for unlabelled boats. A reviewer then
lists, per photo, which numbered people are physically in the water (record_people.py).

  python scripts/assist_people.py --start 0 --count 128 --out reports/people_r3_001
"""
import argparse,csv,json,math,random
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
from stage_candidates import ROOT,jsonl
from release_gate import canonical_hash,digest
PERSON={'Person','Man','Woman','Boy','Girl','Human body'}
# Native objects that may be a target class and must be decided, never silently ignored.
WATCH={'Boat','Surfboard','Canoe','Kayak','Jet ski','Personal watercraft','Paddle','Swan','Duck','Goose','Bird','Dog','Mammal','Animal','Seal','Dolphin','Whale','Fish','Turtle','Barge','Ship','Sailboat','Gondola','Raft','Lifejacket','Ball','Toy','Watercraft','Vehicle'}
try:FONT=ImageFont.truetype('C:/Windows/Fonts/consolab.ttf',16)
except OSError:FONT=ImageFont.load_default()
TW,TH=800,470

def names():
 return {r[0]:r[1] for r in csv.reader(open(ROOT/'research/openimages_boxable_classes.csv',encoding='utf8')) if len(r)>=2}

def pool():
 done={o['id'] for o in jsonl(ROOT/'staging/visual_observations.jsonl')}
 for p in (ROOT/'staging/corrections').glob('*.jsonl'):done|={c['id'] for c in jsonl(p)}
 done|={t['id'] for t in jsonl(ROOT/'staging/people_context_triage.jsonl')}
 out=[]
 for p in sorted((ROOT/'staging/candidates').glob('*.jsonl')):
  for r in jsonl(p):
   if r['id'] not in done and 'person_in_water_status_not_established' in r['issues'] and r['image'].get('pixel_sha256'):out.append(r)
 random.Random(20261027).shuffle(out)
 return out

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--start',type=int,default=0);ap.add_argument('--count',type=int,default=128);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 nm=names();recs=pool()[a.start:a.start+a.count]
 a.out=a.out.resolve()
 if a.out.exists():raise FileExistsError('Use a new evidence directory')
 a.out.mkdir(parents=True)
 from ultralytics import YOLO
 det=YOLO(str(ROOT/'weights/yolov8s.pt'));index=[];page=[]
 def flush():
  n=(len(index)-1)//4
  sheet=Image.new('RGB',(2*TW,2*TH),'#0b111a')
  for j,t in enumerate(page):sheet.paste(t,((j%2)*TW,(j//2)*TH))
  dest=a.out/f'page_{n:04d}.jpg';sheet.save(dest,quality=92);ev=digest(dest.read_bytes())
  for row in index[-len(page):]:row['evidence_path']=str(dest);row['evidence_sha256']=ev
  page.clear()
 for start in range(0,len(recs),16):
  chunk=recs[start:start+16]
  res=det.predict([r['image']['path'] for r in chunk],imgsz=1024,conf=0.3,classes=[8],verbose=False,device=0)
  for r,pr in zip(chunk,res):
   im=Image.open(r['image']['path']).convert('RGB');w,h=im.size;s=min(TW/w,(TH-22)/h)
   t=Image.new('RGB',(TW,TH),'#151d2c');t.paste(im.resize((round(w*s),round(h*s))),(0,22));d=ImageDraw.Draw(t)
   people=[];watch=[]
   for u in r['unresolved_objects']:
    n=nm.get(u['native_label'],u['native_label'])
    if n in PERSON:people.append(u)
    elif n in WATCH:watch.append((n,u))
   people.sort(key=lambda u:(u['native_xyxy'][0],u['native_xyxy'][1]))
   def rect(xy,color,label):
    x1,y1,x2,y2=[v*s for v in xy];y1+=22;y2+=22
    d.rectangle([x1,y1,x2,y2],outline=color,width=2);d.text((x1+2,max(22,y1-17)),label,font=FONT,fill=color,stroke_width=2,stroke_fill='black')
   for n,u in watch:rect(u['native_xyxy'],'white',n)
   hints=[]
   for xy in pr.boxes.xyxy.tolist():
    if not any(nm.get(u['native_label'])=='Boat' and min(xy[2],u['native_xyxy'][2])>max(xy[0],u['native_xyxy'][0]) for u in r['unresolved_objects']):hints.append(xy);rect(xy,'#ffd447','boat?')
   for k,u in enumerate(people,1):rect(u['native_xyxy'],'#ff66c9',str(k))
   slot='ABCD'[len(page)]
   d.text((4,2),f"{slot} {r['id']}  people={len(people)}  other={len(watch)}  boat_hints={len(hints)}",font=FONT,fill='white')
   page.append(t)
   index.append({'id':r['id'],'slot':slot,'record_sha256':canonical_hash(r),'image_sha256':r['image']['sha256'],
                 'people':[u['native_xyxy'] for u in people],'other':[[n,u['native_xyxy']] for n,u in watch],'boat_hints':hints})
   if len(page)==4:flush()
 if page:flush()
 (a.out/'index.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in index),encoding='utf8')
 print(json.dumps({'records':len(index),'pages':math.ceil(len(index)/4),'pool_remaining':len(pool())-a.start-len(index),'out':str(a.out)}))
if __name__=='__main__':main()
