"""Render native human boxes as unresolved person-in-water proposals; never approve."""
import json,hashlib,math
from pathlib import Path
from PIL import Image,ImageOps,ImageDraw,ImageFont
from release_gate import canonical_hash
ROOT=Path(__file__).resolve().parents[1]
F=ImageFont.truetype("C:/Windows/Fonts/consola.ttf",15)
def main():
    rows=[json.loads(s) for s in (ROOT/"staging/candidates/afo.jsonl").read_text().splitlines()]
    obs={json.loads(s)["id"] for s in (ROOT/"staging/visual_observations.jsonl").read_text().splitlines()}
    rs=[r for r in rows if r["id"] not in obs and set(n["class_id"] for n in r["native_annotations"])=={0} and len(r["native_annotations"])<=3]
    rs.sort(key=lambda r:canonical_hash({"id":r["id"],"seed":20261024}))
    rs=rs[:144];out=ROOT/"reports/afo_human_review_001";out.mkdir(exist_ok=False);idx=[]
    for k in range(0,len(rs),6):
        page=Image.new("RGB",(1280,1650),"#152233");d=ImageDraw.Draw(page)
        for j,r in enumerate(rs[k:k+6]):
            n=k+j;x=(j%2)*640;y=(j//2)*550
            im=Image.open(r["image"]["path"]).convert("RGB");w,h=im.size;scale=min(630/w,350/h)
            v=im.resize((round(w*scale),round(h*scale)));page.paste(v,(x,y+36))
            d.text((x+3,y+2),str(n)+" "+r["id"]+" HUMAN? water/aboard",font=F,fill="white")
            proposals=[]
            for i,o in enumerate(r["unresolved_objects"]):
                xy=o["native_xyxy"];proposals.append({"class_id":4,"xyxy":xy,"native_row":o["native_row"]})
                d.rectangle([x+xy[0]*scale,y+36+xy[1]*scale,x+xy[2]*scale,y+36+xy[3]*scale],outline="#ffb020",width=2)
                pad=max(12,(xy[2]-xy[0])*.15,(xy[3]-xy[1])*.15)
                left=max(0,math.floor(xy[0]-pad));top=max(0,math.floor(xy[1]-pad))
                crop=im.crop((left,top,min(w,math.ceil(xy[2]+pad)),min(h,math.ceil(xy[3]+pad))))
                ImageDraw.Draw(crop).rectangle([xy[0]-left,xy[1]-top,xy[2]-left,xy[3]-top],outline="#ffb020",width=1)
                crop=ImageOps.contain(crop,(205,130));page.paste(crop,(x+i*212,y+412));d.text((x+i*212,y+390),str(i)+" HUMAN? native row"+str(o["native_row"]),font=F,fill="#ffb020")
            idx.append({"ordinal":n,"id":r["id"],"record_sha256":canonical_hash(r),"page":str(out/f"page_{k//6:04d}.jpg"),"proposals":proposals})
        page.save(out/f"page_{k//6:04d}.jpg",quality=95)
    (out/"index.jsonl").write_text("".join(json.dumps(q)+"\n" for q in idx),encoding="utf8")
    print(json.dumps({"prepared":len(idx),"approved":0,"path":str(out)}))
if __name__=="__main__":main()
