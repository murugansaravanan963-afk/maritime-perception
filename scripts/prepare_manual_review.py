#!/usr/bin/env python3
"""Render explicit curator-drawn box drafts. Never commits annotations or approvals."""
import argparse,copy,hashlib,json,math
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont,ImageOps
from stage_candidates import ROOT,jsonl
from release_gate import canonical_hash
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
 ap=argparse.ArgumentParser();ap.add_argument("--spec",type=Path,required=True);a=ap.parse_args()
 s=json.loads(a.spec.read_text(encoding="utf8"));out=ROOT/s["out"]
 if out.exists():raise FileExistsError(out)
 idx={e["ordinal"]:e for e in jsonl(ROOT/s["index"])}
 records={r["id"]:r for r in jsonl(ROOT/s["candidates"])}
 patches=[]
 for key,boxes in s["manual_display_boxes"].items():
  ordinal=int(key);e=idx[ordinal];r=copy.deepcopy(records[e["id"]])
  parent=e.get("candidate_record_sha256",e.get("record_sha256"))
  if canonical_hash(r)!=parent:raise ValueError("Stale parent "+r["id"])
  im=Image.open(r["image"]["path"])
  if sha(r["image"]["path"])!=r["image"]["sha256"]:raise ValueError("Image changed")
  sw,sh=ImageOps.contain(im,tuple(s.get("display_limit",[795,590]))).size
  bs=[]
  for c,xy in boxes:
   if type(c) is not int or c not in range(6):raise ValueError("Invalid class")
   coords=[round(xy[j]*(im.width/sw if j%2==0 else im.height/sh),2) for j in range(4)]
   if not (0<=coords[0]<coords[2]<=im.width and 0<=coords[1]<coords[3]<=im.height):raise ValueError((ordinal,coords))
   bs.append({"class_id":c,"xyxy":coords,"annotation_method":"explicit_manual_visible_extent_in_reviewed_full_frame"})
  page=ROOT/s["context_directory"]/("page_%04d.jpg"%(ordinal//int(s.get("context_page_size",4))))
  r["parent_record_sha256"]=parent;r["boxes"]=bs;r["issues"]=[];r["unresolved_objects"]=[]
  if s.get("viewpoint"):
   r["viewpoint"]=s["viewpoint"];r["viewpoint_basis"]=copy.deepcopy(s["viewpoint_basis"])
  r["resolved_native_context"]={"reason":s["reason"],"source_native_annotations_retained":True}
  r["checks_pending"]=["corrected_overlay_confirmation","source_terms","recording_group_verification","near_duplicate_review","independent_split_assignment"]
  r["corrections"]=[{"kind":"manual_full_frame_class_and_extent_repair","reason":s["reason"],"replacement_boxes":bs,"inspection_evidence":str(page),"inspection_evidence_sha256":sha(page),"manual_display_xyxy":boxes,"manual_display_size":[sw,sh]}]
  patches.append({"id":r["id"],"parent_record_sha256":parent,"corrected_record":r,"ordinal":ordinal})
 out.mkdir(parents=True)
 font=ImageFont.truetype("C:/Windows/Fonts/consola.ttf",17);evidence=[]
 for k in range(0,len(patches),4):
  canvas=Image.new("RGB",(1600,1500),"#172432");d=ImageDraw.Draw(canvas)
  for j,p in enumerate(patches[k:k+4]):
   r=p["corrected_record"];x=j%2*800;y=j//2*750;im=Image.open(r["image"]["path"]).convert("RGB");draw=ImageDraw.Draw(im)
   for b in r["boxes"]:draw.rectangle(b["xyxy"],outline=["#ff5b58","#c990ff","#3fe3d5","#ffd447","#ff66c9","#82ee62"][b["class_id"]],width=2)
   shown=ImageOps.contain(im,(795,520));canvas.paste(shown,(x,y+35));d.text((x+4,y+4),str(p["ordinal"])+" "+r["id"],font=font,fill="white")
   tilew=min(390,790//max(1,len(r["boxes"])))
   for bi,b in enumerate(r["boxes"]):
    xy=b["xyxy"];pad=max(8,(xy[2]-xy[0])*.08,(xy[3]-xy[1])*.08)
    crop=im.crop((max(0,math.floor(xy[0]-pad)),max(0,math.floor(xy[1]-pad)),min(im.width,math.ceil(xy[2]+pad)),min(im.height,math.ceil(xy[3]+pad))))
    crop=ImageOps.contain(crop,(tilew-4,178));canvas.paste(crop,(x+bi*tilew,y+563));d.text((x+bi*tilew,y+543),"class"+str(b["class_id"]),font=font,fill="white")
   evidence.append({"ordinal":p["ordinal"],"id":r["id"],"record_sha256":canonical_hash(r),"page":str(out/("page_%04d.jpg"%(k//4)))})
  canvas.save(out/("page_%04d.jpg"%(k//4)),quality=96)
 (out/"draft_corrections.jsonl").write_text("".join(json.dumps(p)+"\n" for p in patches),encoding="utf8")
 (out/"index.jsonl").write_text("".join(json.dumps(p)+"\n" for p in evidence),encoding="utf8")
 (out/"spec.json").write_text(json.dumps(s,indent=2),encoding="utf8")
 print(json.dumps({"drafts":len(patches),"pages":math.ceil(len(patches)/4),"committed":False}))
if __name__=="__main__":main()
