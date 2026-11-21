"""Preserve original buoy classification crops; never invent detection boxes."""
import csv,hashlib,json,zipfile
from collections import Counter
from pathlib import Path
from stage_candidates import make_record,check_image
from inspect_remote_zip import safe_name
ROOT=Path(__file__).resolve().parents[1]
COMMIT="b89e781e502a84c990887df7b8ef71f41dd9bca8"
def main():
    base=ROOT/"raw/iala10";archive=base/("iala10_"+COMMIT[:8]+".zip")
    research=ROOT/"research/expansion_20261024";tree=json.loads((research/"iala10_tree.txt").read_text())
    entries={e["path"]:e for e in tree["tree"] if e["type"]=="blob"}
    provenance=(research/"iala10_provenance.txt").read_bytes();ah=hashlib.sha256(provenance).hexdigest()
    records=[];errors=[]
    with zipfile.ZipFile(archive) as z:
        prefix=z.namelist()[0].split("/")[0]+"/"
        original=z.read(prefix+"dataset/provenance.csv")
        if original!=provenance:raise ValueError("Pinned CSV mismatch")
        meta={r["uuid"]:r for r in csv.DictReader(original.decode("utf-8-sig").splitlines())}
        image_names=[n for n in z.namelist() if n.startswith(prefix+"dataset/") and Path(n).suffix.lower() in [".jpg",".jpeg",".png"]]
        if len(image_names)!=len(meta):raise ValueError("Image/provenance count mismatch")
        for name in image_names:
            rel=name[len(prefix):];native=meta[Path(name).stem];parts=safe_name(rel);data=z.read(name)
            e=entries[rel];blobhash=hashlib.sha1(b"blob "+str(len(data)).encode()+b"\0"+data).hexdigest()
            if len(data)!=e["size"] or blobhash!=e["sha"]:raise ValueError("Git blob mismatch")
            f=base/"original"/Path(*parts);f.parent.mkdir(parents=True,exist_ok=True)
            if f.exists() and f.read_bytes()!=data:raise ValueError("Existing original changed")
            if not f.exists():f.write_bytes(data)
            group="proposal:iala10:source_image:"+native["source"]+":"+native["source_image_id"]
            r=make_record("iala10:"+native["uuid"],"iala10_original",ah,f,source_split="unspecified",group=group,view="unknown")
            r["source_revision"]="github:alemamm/iala10-benchmark:"+COMMIT;r["native_classification"]=native;r["independent_unit"]="iala10:source_image:"+native["source"]+":"+native["source_image_id"]
            r["source_terms"]={"license":"CC-BY-4.0","path":str(research/"iala10_data_license.txt"),"publisher":"https://github.com/alemamm/iala10-benchmark"}
            r["source_image_is_publisher_crop"]=True;r["coordinate_contract"]="no_native_detection_boxes_classification_only"
            r["unresolved_objects"]=[{"native_class":native["class"],"reason":"Classification crop only. Water context, complete target inventory and explicit tight detection boxes require visual annotation."}]
            r["issues"]=["classification_is_not_a_detection_box"]
            r["checks_pending"]+=["water_context_review","manual_detection_boxes"]
            check_image(r);records.append(r)
            if "pixel_sha256" not in r["image"]:errors.append(r["id"])
    output=ROOT/"staging/candidates/iala10.jsonl";text="".join(json.dumps(r,ensure_ascii=False)+"\n" for r in sorted(records,key=lambda r:r["id"]))
    if output.exists() and output.read_text(encoding="utf8")!=text:raise ValueError("Candidate manifest changed")
    output.write_text(text,encoding="utf8")
    report={"state":"PUBLISHER_CLASSIFICATION_CROPS_STAGED_NOT_DETECTION_APPROVED","images":len(records),"decode_failures":errors,"distinct_decoded_pixels":len({r["image"].get("pixel_sha256") for r in records}),"distinct_original_source_image_groups":len({r["capture_group"] for r in records}),"native_class_counts":dict(Counter(r["native_classification"]["class"] for r in records)),"images_minimum_side_under96":sum(min(r["image"]["width"],r["image"]["height"])<96 for r in records),"archive_sha256":hashlib.sha256(archive.read_bytes()).hexdigest(),"git_commit":COMMIT,"native_csv_sha256":ah,"manifest_sha256":hashlib.sha256(output.read_bytes()).hexdigest(),"release_approvals":0}
    (ROOT/"reports/iala10_acquisition.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8");print(json.dumps(report,indent=2))
if __name__=="__main__":main()
