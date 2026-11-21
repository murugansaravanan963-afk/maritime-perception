"""Stage original wet-wood RGB frames, excluding dry/derived views; no release."""
import json,zipfile,hashlib,collections,io
from pathlib import Path
from PIL import Image
from inspect_remote_zip import safe_name
from stage_candidates import make_record,check_image
from audit_candidates import box_issues
ROOT=Path(__file__).resolve().parents[1]
def main():
    base=ROOT/"raw/instream_wood_expansion";archive=base/"extra_own_data.zip"
    receipt=json.loads((base/"extra_own_data.zip.receipt.json").read_text());md=hashlib.md5()
    with archive.open("rb") as f:
        for block in iter(lambda:f.read(8*1048576),b""):md.update(block)
    assert md.hexdigest()==receipt["md5"]=="9f69d822a6480af815a01957c42d4145"
    groups={"VDN_vid_20210509_174821":"VDN_site_20210509","VDN_vid_20210509_180320":"VDN_site_20210509","20220520_rhone_data":"Rhone_20220520","202306_Spoel_data_Inn":"Inn_202306"}
    rows=[];inventory=[];names_seen=set()
    with zipfile.ZipFile(archive) as z:
        labels={(n.split("/")[1],Path(n).stem):n for n in z.namelist() if n.endswith(".txt")}
        for n in z.namelist():
            if not n.endswith(".jpg"):continue
            parts=safe_name(n);group=parts[1]
            reason="dry_wood_outside_water_target" if group.startswith("dry_wood") else "derived_merged_channels_not_an_independent_original" if "/jpgs_merged/" in n else None
            if reason:inventory.append({"member":n,"excluded":reason});continue
            assert group in groups and "/jpgs/" in n
            key=(group,Path(n).stem);assert key not in names_seen;names_seen.add(key)
            assert key in labels,"Missing original annotation"
            data=z.read(n);annotation=z.read(labels[key]);ah=hashlib.sha256(annotation).hexdigest()
            f=base/"original"/Path(*parts);a=base/"original"/Path(*safe_name(labels[key]))
            for p,b in [(f,data),(a,annotation)]:
                p.parent.mkdir(parents=True,exist_ok=True)
                if p.exists() and p.read_bytes()!=b:raise ValueError("Original changed")
                if not p.exists():p.write_bytes(b)
            im=Image.open(io.BytesIO(data));w,h=im.size
            native=[]
            for i,line in enumerate(annotation.decode("utf-8-sig").splitlines()):
                if not line.strip():continue
                v=line.split()
                if len(v)!=5 or v[0]!="0":raise ValueError("Unknown native wood schema")
                cx,cy,bw,bh=map(float,v[1:]);xy=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h]
                native.append({"native_class":0,"xyxy":xy,"native_yolo":list(map(float,v[1:])),"native_row":i+1})
            r=make_record("wood_extra:"+group+":"+Path(n).stem,"instream_wood_extra",ah,f,source_split="unspecified",group="proposal:wood_extra:"+groups[group],view="shore")
            r["source_revision"]="zenodo:10869058:extra_own_data.zip:md5:"+receipt["md5"]
            r["source_annotation_path"]=str(a);r["native_annotations"]=native;r["native_capture_folder"]=group
            r["boxes"]=[{"class_id":5,"xyxy":o["xyxy"],"native_class":"wood","native_row":o["native_row"]} for o in native]
            r["issues"]+=box_issues(native,w,h)
            if not native:r["issues"].append("no_native_positive_label_full_frame_review_needed")
            r["coordinate_contract"]="publisher_normalized_YOLO_center_xywh_converted_without_clipping"
            r["source_terms"]={"license":"CC-BY-4.0","publisher":"https://zenodo.org/records/10869058","evidence":str(ROOT/"research/instream_wood_latest.json")}
            r["acquisition_receipt"]={"archive_sha256":receipt["sha256"],"archive_member":n,"annotation_member":labels[key],"original_bytes_preserved":True}
            r["checks_pending"]+=["full_frame_unlabelled_people_animals_craft_review","floating_not_attached_or_dry_wood_review","capture_event_temporal_thinning"]
            check_image(r);rows.append(r)
            inventory.append({"member":n,"annotation_member":labels[key],"id":r["id"],"image_sha256":r["image"]["sha256"],"native_geometry_issues":r["issues"]})
    out=ROOT/"staging/candidates/wood_extra.jsonl";data="".join(json.dumps(q)+"\n" for q in sorted(rows,key=lambda q:q["id"]))
    if out.exists() and out.read_text()!=data:raise ValueError("Existing candidate inventory changed")
    out.write_text(data,encoding="utf8")
    report={"state":"ORIGINAL_WET_WOOD_FRAMES_STAGED_NOT_APPROVED","staged":len(rows),"decoded":sum("pixel_sha256" in r["image"] for r in rows),"original_native_capture_folders":dict(collections.Counter(r["native_capture_folder"] for r in rows)),"excluded_reason_counts":dict(collections.Counter(i["excluded"] for i in inventory if "excluded" in i)),"structural_holds":sum(bool(r["issues"]) for r in rows),"empty_native_labels":sum(not r["boxes"] for r in rows),"candidate_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"archive_receipt":receipt,"release_approvals":0}
    (ROOT/"reports/wood_extra_acquisition.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    (ROOT/"research/expansion_20261024/wood_extra_staging_inventory.jsonl").write_text("".join(json.dumps(i)+"\n" for i in inventory),encoding="utf8")
    print(json.dumps(report,indent=2))
if __name__=="__main__":main()
