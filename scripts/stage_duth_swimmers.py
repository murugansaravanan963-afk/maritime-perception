"""Stage original publisher TRAIN images, preserving every annotation for visual review."""
import json,zipfile,hashlib,math,re,collections
from pathlib import Path
from PIL import Image
from stage_candidates import ROOT,make_record,check_image
from release_gate import canonical_hash

def main():
 archive=ROOT/"raw/duth_swimmers/swimmers_dataset.zip"
 inv=json.loads((ROOT/"research/expansion_20261025/duth_swimmers_archive_inventory.json").read_text(encoding="utf8"))
 if hashlib.sha256(archive.read_bytes()).hexdigest()!=inv["sha256"]:raise ValueError("Archive changed")
 report=json.loads((ROOT/"research/expansion_20261025/duth_swimmers_annotation_inventory.json").read_text(encoding="utf8"))
 output=ROOT/"staging/candidates/duth_swimmers.jsonl"
 if output.exists():raise FileExistsError(output)
 out=ROOT/"raw/duth_swimmers/original_training";out.mkdir(exist_ok=False)
 records=[];unmatched=[];ambiguous=[];sizes=collections.Counter()
 with zipfile.ZipFile(archive) as z:
  members={n for n in z.namelist() if not n.endswith("/")}
  for k,label in enumerate(sorted(report["positive_label_names"]),1):
   stem=Path(label).stem
   images=[n for n in members if Path(n).stem==stem and Path(n).suffix.lower() in [".jpg",".jpeg",".png"]]
   if len(images)!=1:
    (unmatched if not images else ambiguous).append({"label":label,"images":images});continue
   name=images[0];data=z.read(name);lb=z.read(label)
   ip=out/Path(name).name;lp=out/Path(label).name
   if ip.exists() or lp.exists():raise ValueError("Name collision")
   ip.write_bytes(data);lp.write_bytes(lb)
   with Image.open(ip) as im:w,h=im.size
   r=make_record("duth_swimmers:"+stem,"duth_swimmers_original",hashlib.sha256(lb).hexdigest(),ip,
     source_split="train",group="campaign:duth_swimmers:original_training_collection",view="unknown")
   r["source_revision"]="publisher_archive_sha256:"+inv["sha256"]
   r["native_label_path"]=str(lp);r["native_image_member"]=name;r["native_label_member"]=label
   r["coordinate_contract"]="native_YOLO_normalized_center_xywh_to_zero_origin_continuous_xyxy"
   r["native_annotations"]=[]
   for line in lb.decode("utf-8-sig").splitlines():
    if not line.strip():continue
    v=line.split();c=int(v[0]);cx,cy,bw,bh=map(float,v[1:])
    if c!=0:raise ValueError("Unexpected native class")
    if not all(math.isfinite(x) for x in [cx,cy,bw,bh]):raise ValueError("Non-finite")
    xy=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]
    r["native_annotations"].append({"native_class":c,"normalized_center_xywh":[cx,cy,bw,bh],"xyxy":xy})
    r["boxes"].append({"class_id":4,"xyxy":xy,"native_class":"swimmer","mapping_status":"provisional_until_full_frame_visual_review"})
   r["source_terms"]={"publisher":"https://robotics.pme.duth.gr/research/swimmers","evidence_path":str(ROOT/"research/expansion_20261025/duth_swimmers_landing.txt"),"basis":"Publisher explicitly freely distributes annotated swimmer images for researchers training water search-and-rescue detectors; retain citations. No commercial or public redistribution permission inferred."}
   r["checks_pending"]+=["native_swimmer_context_and_full_extent","source_collection_event_identity","transformation_and_cross_source_duplicates","viewpoint_confirmation"]
   r["capture_group_basis"]="Whole publisher collection conservatively grouped because filenames alone do not establish independent events. Alphabetic prefixes are not assumed independent or original."
   r=check_image(r,(w,h));records.append(r);sizes[(w,h)]+=1
   if k%500==0:print("Staged",len(records),"/",len(report["positive_label_names"]),flush=True)
 output.write_text("".join(json.dumps(r)+"\n" for r in records),encoding="utf8")
 status={"state":"ORIGINAL_TRAIN_CANDIDATES_NOT_RELEASED","records":len(records),"unmatched_labels":unmatched,"ambiguous_image_pairs":ambiguous,"decode_failures":sum(not r["image"].get("pixel_sha256") for r in records),"geometry_or_decode_issues":sum(bool(r["issues"]) for r in records),"distinct_decoded_pixels":len({r["image"].get("pixel_sha256") for r in records}),"native_positive_images_per_prefix":dict(collections.Counter(re.sub(r"\d.*$","",Path(r["native_image_member"]).stem) or "numeric" for r in records)),"resolution_counts":{str(k):v for k,v in sizes.items()},"manifest_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"archive_sha256":inv["sha256"],"automatic_approvals":0,"nested_archive_opened":False}
 (ROOT/"reports/duth_swimmers_acquisition.json").write_text(json.dumps(status,indent=2),encoding="utf8")
 print(json.dumps({k:v for k,v in status.items() if k!="resolution_counts"},indent=2),flush=True)
if __name__=="__main__":main()
