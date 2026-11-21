"""Acquire only native training-positive WaterScenes RGB members, never test payloads."""
import json,zipfile,hashlib,copy,csv,time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
import xml.etree.ElementTree as ET
from stage_candidates import make_record,check_image
from audit_candidates import box_issues
from remote_zip_member import fetch_member
ROOT=Path(__file__).resolve().parents[1]
def main():
    root=ROOT/"raw/waterscenes";idx=json.loads((ROOT/"research/expansion_20261024/waterscenes_image_archive_index.json").read_text())
    entries={Path(e["name"]).stem:e for e in idx["entries"] if e["name"].endswith(".jpg")}
    inventory=[json.loads(s) for s in (ROOT/"reports/waterscenes_train_inventory.jsonl").read_text().splitlines()]
    selected=[r for r in inventory if not r["ambiguous_vessel_frame"] and set(r["native_classes"])&{"buoy","kayak"}]
    train={Path(s).stem for s in (root/"train.txt").read_text().splitlines()}
    test={Path(s).stem for s in (root/"test.txt").read_text().splitlines()}
    val={Path(s).stem for s in (root/"val.txt").read_text().splitlines()}
    assert not train.intersection(test|val)
    scene={r["id"]:r for r in csv.DictReader((root/"information_list.csv").open(newline="",encoding="utf-8-sig"))}
    tasks=[];excluded=[]
    with zipfile.ZipFile(root/"detection.zip") as z:
        for r in selected:
            rid=r["id"];assert rid in train and rid not in test|val
            payload=z.read("detection/xml/"+rid+".xml")
            if hashlib.sha256(payload).hexdigest()!=r["annotation_sha256"]:raise ValueError("Native XML changed")
            x=ET.fromstring(payload);w=int(x.findtext("size/width"));h=int(x.findtext("size/height"))
            native=[{"native_class":o.findtext("name"),"xyxy":[float(o.findtext("bndbox/"+k)) for k in ["xmin","ymin","xmax","ymax"]],"difficult":o.findtext("difficult")} for o in x.findall("object")]
            errors=box_issues(native,w,h)
            if errors:excluded.append({"id":rid,"issues":errors});continue
            assert r["original_filename"]==x.findtext("filename")
            e=entries[rid]
            e={"name":e["name"],"bytes":e["size"],"compressed_bytes":e["compressed_size"],"crc32":format(e["crc32"],"08x"),"header_offset":e["header_offset"]}
            tasks.append((r,payload,w,h,native,e))
    folder=root/"images_train";folder.mkdir(exist_ok=True)
    (root/"xml_train").mkdir(exist_ok=True)
    receipts=root/"image_receipts.jsonl";known={q["id"]:q for q in map(json.loads,receipts.read_text().splitlines())} if receipts.exists() else {}
    def fetch(task):
        r,xml,w,h,native,e=task;rid=r["id"];f=folder/(rid+".jpg")
        if f.exists():
            data=f.read_bytes()
            if rid not in known or hashlib.sha256(data).hexdigest()!=known[rid]["image_sha256"]:raise ValueError("Existing original lacks matching receipt")
        else:
            for attempt in range(3):
                try:data=fetch_member(idx["url"],idx["size"],e);break
                except Exception:
                    if attempt==2:raise
                    time.sleep(2+attempt*2)
            f.write_bytes(data)
        sh=hashlib.sha256(data).hexdigest()
        rec={"id":rid,"image_sha256":sh,"archive_member":e,"archive_index_sha256":hashlib.sha256((ROOT/"research/expansion_20261024/waterscenes_image_archive_index.json").read_bytes()).hexdigest()}
        return task,rec
    done=[];failures=[]
    with ThreadPoolExecutor(max_workers=6) as pool:
        fs={pool.submit(fetch,t):t[0]["id"] for t in tasks}
        for future in as_completed(fs):
            try:
                task,receipt=future.result();done.append((task,receipt))
                if receipt["id"] not in known:
                    with receipts.open("a",encoding="utf8") as f:f.write(json.dumps(receipt)+"\n")
                if len(done)%100==0:print(json.dumps({"acquired_or_verified":len(done),"selected":len(tasks),"failures":len(failures)}),flush=True)
            except Exception as ex:failures.append({"id":fs[future],"error":type(ex).__name__+": "+str(ex)});print(json.dumps(failures[-1]),flush=True)
    rows=[]
    for task,receipt in done:
        meta,xml,w,h,native,e=task;rid=meta["id"]
        xf=root/"xml_train"/(rid+".xml")
        if xf.exists() and xf.read_bytes()!=xml:raise ValueError("Existing native XML changed")
        xf.write_bytes(xml)
        stamp=meta["original_filename"].split(".")[0]
        # Timestamp/date alone is a proposal, not independence certification.
        from datetime import datetime,timezone
        day=datetime.fromtimestamp(int(stamp),timezone.utc).date().isoformat()
        r=make_record("waterscenes:"+rid,"waterscenes_original",meta["annotation_sha256"],folder/(rid+".jpg"),source_split="train",group="proposal:waterscenes:date:"+day,view="boat")
        r["source_revision"]="github:WaterScenes/WaterScenes:9b6af76c4060cf523398f97dc0992ff314791dcc"
        r["source_annotation_path"]=str(xf);r["native_annotations"]=native;r["native_original_filename"]=meta["original_filename"];r["native_scene_metadata"]=scene[rid]
        r["coordinate_contract"]="publisher_XML_direct_xyxy_matches_publisher_normalized_YOLO_without_VOC_minus1"
        for i,o in enumerate(native):
            name=o["native_class"]
            if name in ["buoy","kayak"]:r["boxes"].append({"class_id":3 if name=="buoy" else 2,"xyxy":o["xyxy"],"native_class":name,"native_row":i+1})
            elif name=="pier":pass
            else:r["unresolved_objects"].append({**o,"reason":"Whole-frame semantic review required; native sailor is not automatically a swimmer."})
        if r["unresolved_objects"]:r["issues"].append("native_person_or_vessel_context_unresolved")
        r["resolved_native_context"]={"ignored_pier":"Native piers are permanent structures outside the six target classes; this does not certify other missing targets."}
        r["source_terms"]={"license":"CC-BY-NC-SA-4.0","publisher":"https://github.com/WaterScenes/WaterScenes","evidence":str(ROOT/"research/WaterScenes__WaterScenes__README.md")}
        r["acquisition_receipt"]=receipt;r["checks_pending"]+=["native_box_visible_extent_review","full_frame_missing_targets_review","capture_timestamp_temporal_thinning"]
        check_image(r)
        if r["image"].get("width")!=w or r["image"].get("height")!=h:r["issues"].append("native_image_dimensions_mismatch")
        rows.append(r)
    rows.sort(key=lambda r:r["id"])
    out=ROOT/"staging/candidates/waterscenes.jsonl";data="".join(json.dumps(r)+"\n" for r in rows)
    if out.exists() and out.read_text()!=data:raise ValueError("Candidate manifest already exists and changed")
    out.write_text(data,encoding="utf8")
    report={"state":"NATIVE_TRAIN_RGB_ACQUIRED_NOT_APPROVED","selected":len(selected),"native_geometry_excluded":excluded,"acquired_staged":len(rows),"failures":failures,"decoded":sum("pixel_sha256" in r["image"] for r in rows),"publisher_test_images_opened":0,"publisher_validation_images_opened":0,"publisher_test_annotations_opened":0,"original_photo_members_only":True,"candidate_manifest_sha256":hashlib.sha256(out.read_bytes()).hexdigest(),"release_approvals":0}
    (ROOT/"reports/waterscenes_acquisition.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8")
    print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
