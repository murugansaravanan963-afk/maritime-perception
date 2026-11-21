"""Acquire original publisher TRAIN AFO images with verified ZIP member CRCs."""
import argparse,hashlib,json,time,zlib
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from stage_candidates import make_record,check_image
from remote_zip_member import fetch_member
from release_gate import canonical_hash
ROOT=Path(__file__).resolve().parents[1]
NAMES=["human","wind/sup-board","boat","bouy","sailboat","kayak"]
MAPPING={3:3,4:1,5:2}
def rows(p):return [json.loads(s) for s in p.read_text(encoding="utf8").splitlines() if s.strip()]
def main():
    ap=argparse.ArgumentParser();ap.add_argument("--max-boxes",type=int,default=6);ap.add_argument("--workers",type=int,default=6);a=ap.parse_args()
    research=ROOT/"research/expansion_20261024";ix=json.loads((research/"afo_archive_index.json").read_text())
    summary=json.loads((research/"afo_train_inventory_summary.json").read_text());bad={r[0] for r in summary["native_geometry_issues"]}
    selected=[r for r in rows(research/"afo_train_annotation_inventory.jsonl") if r["name"] not in bad and 0<len(r["boxes"])<=a.max_boxes]
    # Prioritize underrepresented kayak examples, then board examples, then other eligible training scenes.
    selected.sort(key=lambda r:(0 if any(b["class_id"]==5 for b in r["boxes"]) else 1 if any(b["class_id"]==1 for b in r["boxes"]) else 2,r["name"]))
    entries={Path(e["name"]).name:e for e in ix["entries"] if Path(e["name"]).suffix.lower() in [".jpg",".png",".jpeg"]}
    if len(entries)!=sum(Path(e["name"]).suffix.lower() in [".jpg",".png",".jpeg"] for e in ix["entries"]):raise ValueError("Ambiguous native image names")
    base=ROOT/"raw/afo";dest=base/"images_train";dest.mkdir(exist_ok=True)
    path=base/"image_receipts.jsonl";previous={r["name"]:r for r in rows(path)} if path.exists() else {}
    receipts={};failures=[]
    def fetch(r):
        e=entries[r["name"]];f=dest/r["name"]
        if f.exists():data=f.read_bytes()
        else:
            error=None
            for attempt in range(3):
                try:data=fetch_member(ix["resolved_archive_url"],ix["archive_bytes"],e);error=None;break
                except Exception as ex:error=ex;time.sleep(min(2**attempt,4))
            if error:raise error
            temp=f.with_suffix(f.suffix+".part");temp.write_bytes(data);temp.replace(f)
        if len(data)!=e["bytes"] or format(zlib.crc32(data)&0xffffffff,"08x")!=e["crc32"]:raise ValueError("Source member CRC/length mismatch")
        d=hashlib.sha256(data).hexdigest()
        if r["name"] in previous and d!=previous[r["name"]]["sha256"]:raise ValueError("Original image changed")
        return {"name":r["name"],"path":str(f),"member":e["name"],"bytes":len(data),"sha256":d,"crc32":e["crc32"],"publisher_split":"train"}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs={pool.submit(fetch,r):r for r in selected}
        for n,future in enumerate(as_completed(jobs),1):
            r=jobs[future]
            try:
                rec=future.result();receipts[r["name"]]=rec
                with path.open("a",encoding="utf8") as f:f.write(json.dumps(rec)+"\n")
            except Exception as ex:failures.append({"name":r["name"],"error":str(ex)})
            if n%50==0:print(json.dumps({"acquired_or_verified":n,"selected":len(selected),"failures":len(failures)}),flush=True)
    allreceipts={**previous,**receipts};path.write_text("".join(json.dumps(r)+"\n" for _,r in sorted(allreceipts.items())),encoding="utf8")
    records=[]
    for src in selected:
        if src["name"] not in receipts:continue
        receipt=receipts[src["name"]];prefix=Path(src["name"]).stem.rsplit("_",1)[0]
        r=make_record("afo:"+Path(src["name"]).stem,"afo_original_v1",src["annotation_sha256"],Path(receipt["path"]),source_split="train",group="proposal:afo:filename_prefix:"+prefix,view="aerial")
        r["source_revision"]="publisher_kaggle:1381740:version1";r["native_annotations"]=src["boxes"];r["source_annotation_path"]=src["annotation_path"];r["native_class_names"]=NAMES
        r["source_terms"]={"license_as_dataset_readme":"CC-BY-NC-SA-3.0","publisher_metadata_license_label":"CC-BY-NC-SA-3.0-IGO","evidence":str(base/"metadata/readme.txt"),"publisher":"https://www.kaggle.com/datasets/jangsienicajzkowy/afo-aerial-dataset-of-floating-objects","note":"Publisher label and prose variants preserved; do not infer commercial or unrestricted redistribution rights."}
        r["coordinate_contract"]="native_Darknet_normalized_center_xywh_to_continuous_pixel_xyxy";r["checks_pending"]+=["human_water_contact_or_aboard_review","wind_vs_sup_board_review","native_craft_extent_with_occupants"]
        check_image(r);w,h=r["image"]["width"],r["image"]["height"]
        for b in src["boxes"]:
            c=b["class_id"];x,y,bw,bh=b["native_yolo"];xy=[(x-bw/2)*w,(y-bh/2)*h,(x+bw/2)*w,(y+bh/2)*h]
            if c in MAPPING:r["boxes"].append({"class_id":MAPPING[c],"xyxy":xy,"native_class":NAMES[c],"native_row":b["native_row"]})
            else:r["unresolved_objects"].append({"native_class":NAMES[c],"native_class_id":c,"native_row":b["native_row"],"native_xyxy":xy})
        if r["unresolved_objects"]:r["issues"].append("native_context_class_not_resolved")
        from audit_candidates import box_issues
        r["issues"]+=box_issues([{"native_class":b["class_id"],"xyxy":b["xyxy"]} for b in r["boxes"]],w,h)
        records.append(r)
    output=ROOT/"staging/candidates/afo.jsonl"
    new="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in records)
    if output.exists() and output.read_text(encoding="utf8")!=new:raise FileExistsError("Existing source manifest differs; inspect before replacing")
    output.write_text(new,encoding="utf8")
    report={"state":"ORIGINAL_TRAIN_IMAGES_ACQUIRED_NOT_APPROVED","selected":len(selected),"decoded_staged":len(records),"failures":failures,"publisher_test_images_opened":0,"publisher_test_labels_opened":0,"native_train_images_excluded_for_geometry":len(bad),"selected_native_positive_images":dict(Counter(c for r in selected for c in {b["class_id"] for b in r["boxes"]})),"classes":NAMES,"candidate_manifest":str(output),"candidate_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"release_approvals":0}
    (ROOT/"reports/afo_acquisition.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8");print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
