"""Evidence sheets showing full frames plus every mapped object, no auto approvals."""
import argparse,json,math,random,re
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw,ImageFont
from stage_candidates import ROOT,jsonl
from release_gate import canonical_hash,NAMES
from render_review import COLORS
F=ImageFont.truetype('C:/Windows/Fonts/consola.ttf',15)
def main():
 p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--out',required=True);p.add_argument('--limit',type=int,default=72);p.add_argument('--offset',type=int,default=0);p.add_argument('--max-boxes',type=int,default=3);p.add_argument('--class-id',type=int);p.add_argument('--require-current-unblocked',action='store_true');p.add_argument('--outline-crops',action='store_true');p.add_argument('--id-regex',help='Optional id pattern for a declared review subset; not a class approval');a=p.parse_args()
 obs={r['id'] for r in jsonl(ROOT/'staging/visual_observations.jsonl')}
 eligible={r['id'] for r in jsonl(ROOT/'staging/candidate_master.jsonl') if not r['issues']} if a.require_current_unblocked else None
 rs=[r for r in jsonl(ROOT/f'staging/candidates/{a.source}.jsonl') if not r['issues'] and not r['unresolved_objects'] and 0<len(r['boxes'])<=a.max_boxes and r['id'] not in obs and (eligible is None or r['id'] in eligible) and (a.class_id is None or any(b['class_id']==a.class_id for b in r['boxes']))]
 if a.id_regex:rs=[r for r in rs if re.search(a.id_regex,r['id'])]
 # Fixed random order spreads review across groups; not a split assignment.
 rs.sort(key=lambda r:canonical_hash({'id':r['id'],'seed':20261023}));rs=rs[a.offset:a.offset+a.limit]
 out=ROOT/'reports'/a.out
 if out.exists():raise FileExistsError('Evidence directory already exists; choose a new --out: '+str(out))
 out.mkdir(parents=True);index=[]
 for base in range(0,len(rs),6):
  page=Image.new('RGB',(1280,1650),'#152233');dr=ImageDraw.Draw(page)
  for j,r in enumerate(rs[base:base+6]):
   num=a.offset+base+j;x=j%2*640;y=j//2*550;im=Image.open(r['image']['path']).convert('RGB');w,h=im.size;s=min(630/w,350/h);v=im.resize((round(w*s),round(h*s)));page.paste(v,(x,y+36));d=ImageDraw.Draw(page)
   d.text((x+3,y+2),str(num)+' '+r['id'],font=F,fill='white')
   for bi,b in enumerate(r['boxes']):
    xy=b['xyxy'];c=COLORS[b['class_id']];z=[x+xy[0]*s,y+36+xy[1]*s,x+xy[2]*s,y+36+xy[3]*s];d.rectangle(z,outline=c,width=2);d.text((z[0],max(y+36,z[1]-16)),str(bi),font=F,fill=c,stroke_width=1,stroke_fill='black')
    pad=max(8,(xy[2]-xy[0])*.12,(xy[3]-xy[1])*.12);left=max(0,math.floor(xy[0]-pad));top=max(0,math.floor(xy[1]-pad));crop=im.crop((left,top,min(w,math.ceil(xy[2]+pad)),min(h,math.ceil(xy[3]+pad))));
    if a.outline_crops:ImageDraw.Draw(crop).rectangle([xy[0]-left,xy[1]-top,xy[2]-left,xy[3]-top],outline=c,width=1)
    crop=ImageOps.contain(crop,(205,130));page.paste(crop,(x+bi*212,y+412));d.text((x+bi*212,y+390),str(bi)+' '+NAMES[b['class_id']],font=F,fill=c)
   index.append({'ordinal':num,'id':r['id'],'record_sha256':canonical_hash(r),'page':str(out/f'page_{(a.offset+base)//6:04d}.jpg'),'image_sha256':r['image']['sha256']})
  page.save(out/f'page_{(a.offset+base)//6:04d}.jpg',quality=95)
 (out/'index.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in index),encoding='utf8');print(json.dumps({'frames':len(index),'path':str(out)}))
if __name__=='__main__':main()
