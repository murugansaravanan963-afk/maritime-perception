"""Acquire original MCAD RGB TRAIN members with native labels; no approvals."""
import json,sys,zipfile,hashlib,zlib,math,time
from pathlib import Path
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from inspect_remote_zip import HTTPRangeFile,safe_name
from remote_zip_member import fetch_member
from stage_candidates import make_record,check_image
from audit_candidates import box_issues
ROOT=Path(__file__).resolve().parents[1]
NAMES=["obstacle","power-driven vessel","sailboat","seamark","seaplane"]
MAPPING={1:0,2:1,3:3}
def main():
    rd=ROOT/"research/expansion_20261024_addendum"
    ix=json.loads((rd/"mcad_archive_index.json").read_text(encoding="utf8"))
    entries={e["name"]:e for e in ix["entries"]}
    if len(entries)!=len(ix["entries"]):raise ValueError("Duplicate archive names")
    prefix="maritime_collision_avoidance_de_gb_nl/rgb/"
    ims=[e for e in ix["entries"] if e["name"].startswith(prefix+"images/train/") and e["name"].endswith(".jpg")]
    base=ROOT/"raw/mcad_rgb";annout=base/"labels_train";dest=base/"images_train"
    annout.mkdir(parents=True,exist_ok=True);dest.mkdir(exist_ok=True)
    inventory=[];st=HTTPRangeFile(ix["url"],ix["archive_bytes"])
    # Block cache may fetch bytes from adjacent archive entries. Only metadata and
    # the label matching an original TRAIN image are extracted/parsed. No held-out
    # image or label entry is opened, decoded or used to select examples.
    with zipfile.ZipFile(st) as z:
        for e in ims:
            stem=Path(e["name"]).stem;name=prefix+"labels/train/"+stem+".txt"
            if name not in entries:raise ValueError("Missing original TRAIN label "+stem)
            data=z.read(name);target=annout/(stem+".txt")
            if target.exists()and target.read_bytes()!=data:raise ValueError("Label source changed")
            target.write_bytes(data);boxes=[];errors=[]
            for line_no,line in enumerate(data.decode("utf-8-sig").splitlines(),1):
                v=line.split()
                if len(v)!=5:raise ValueError("Invalid native row "+stem)
                cid=int(v[0]);a=list(map(float,v[1:]))
                if cid not in range(5)or any(not math.isfinite(x)for x in a):raise ValueError("Invalid native class/float")
                cx,cy,w,h=a;xy=[cx-w/2,cy-h/2,cx+w/2,cy+h/2]
                if not(0<=xy[0]<xy[2]<=1 and 0<=xy[1]<xy[3]<=1):errors.append("native_normalized_geometry_invalid")
                boxes.append({"native_class_id":cid,"native_class":NAMES[cid],"native_row":line_no,"native_yolo":a})
            inventory.append({"id":stem,"image_entry":e,"annotation_member":name,"annotation_sha256":hashlib.sha256(data).hexdigest(),"annotation_path":str(target),"boxes":boxes,"issues":sorted(set(errors))})
    invpath=rd/"mcad_rgb_train_inventory.jsonl"
    invtext="".join(json.dumps(r)+"\n"for r in inventory)
    if invpath.exists()and invpath.read_text(encoding="utf8")!=invtext:raise ValueError("Inventory changed")
    invpath.write_text(invtext,encoding="utf8")
    selected=[r for r in inventory if not r["issues"] and 0<len(r["boxes"])<=6]
    receipts_path=base/"image_receipts.jsonl"
    previous={r["id"]:r for r in map(json.loads,receipts_path.read_text(encoding="utf8").splitlines())}if receipts_path.exists()else{}
    failures=[];done={}
    def fetch(r):
        e=r["image_entry"];p=dest/(r["id"]+".jpg")
        if p.exists():data=p.read_bytes()
        else:
            error=None
            for attempt in range(3):
                try:data=fetch_member(ix["url"],ix["archive_bytes"],e);error=None;break
                except Exception as ex:error=ex;time.sleep(min(2**attempt,4))
            if error:raise error
            temp=p.with_suffix(".part");temp.write_bytes(data);temp.replace(p)
        if len(data)!=e["bytes"]or format(zlib.crc32(data)&0xffffffff,"08x")!=e["crc32"]:raise ValueError("Original CRC/length mismatch")
        sha=hashlib.sha256(data).hexdigest()
        if r["id"]in previous and previous[r["id"]]["sha256"]!=sha:raise ValueError("Existing image changed")
        return {"id":r["id"],"path":str(p),"member":e["name"],"sha256":sha,"bytes":len(data),"crc32":e["crc32"],"publisher_split":"train"}
    with ThreadPoolExecutor(max_workers=6)as pool:
        jobs={pool.submit(fetch,r):r for r in selected}
        for n,f in enumerate(as_completed(jobs),1):
            r=jobs[f]
            try:
                v=f.result();done[r["id"]]=v
                with receipts_path.open("a",encoding="utf8")as fp:fp.write(json.dumps(v)+"\n")
            except Exception as e:failures.append({"id":r["id"],"error":str(e)})
            if n%50==0:print(json.dumps({"acquired_or_verified":n,"selected":len(selected),"failures":len(failures)}),flush=True)
    receipts={**previous,**done}
    receipts_path.write_text("".join(json.dumps(r)+"\n"for _,r in sorted(receipts.items())),encoding="utf8")
    records=[]
    for src in selected:
        if src["id"]not in done:continue
        v=done[src["id"]]
        r=make_record("mcad_rgb:"+src["id"],"mcad_rgb_original",src["annotation_sha256"],Path(v["path"]),group="campaign:mcad:original_rgb_acquisition",view="boat")
        r["source_revision"]="publisher_archive_last_modified_2025-08-27:1159342014"
        r["source_annotation_path"]=src["annotation_path"];r["native_annotations"]=src["boxes"]
        r["native_class_names"]=NAMES;r["native_member"]=v["member"];r["native_annotation_member"]=src["annotation_member"]
        r["coordinate_contract"]="native_YOLO_normalized_center_xywh_to_continuous_pixel_xyxy"
        r["source_terms"]={"license":"CC-BY-4.0","publisher":"Federal Maritime and Hydrographic Agency (BSH)","url":"https://doi.bsh.de/10.60751/z1w9-t249","evidence":str(rd/"mcad_feed.xml"),"evidence_sha256":hashlib.sha256((rd/"mcad_feed.xml").read_bytes()).hexdigest()}
        r["capture_group_basis"]={"reason":"UUID filenames and camera-angle metadata do not identify recording/day. Protect entire RGB acquisition as one group until actual event relationships are recovered.","source_metadata":str(rd/"mcad_rgb_metadata.txt"),"metadata_sha256":hashlib.sha256((rd/"mcad_rgb_metadata.txt").read_bytes()).hexdigest(),"independence_verified":False}
        check_image(r);w,h=r["image"]["width"],r["image"]["height"]
        for b in src["boxes"]:
            cx,cy,bw,bh=b["native_yolo"];xy=[(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h];c=b["native_class_id"]
            if c in MAPPING:r["boxes"].append({"class_id":MAPPING[c],"xyxy":xy,"native_class":NAMES[c],"native_row":b["native_row"]})
            else:r["unresolved_objects"].append({"native_class":NAMES[c],"native_class_id":c,"native_xyxy":xy,"native_row":b["native_row"]})
        if r["unresolved_objects"]:r["issues"].append("native_context_class_not_resolved")
        r["issues"]+=box_issues([{"native_class":b["class_id"],"xyxy":b["xyxy"]}for b in r["boxes"]],w,h)
        r["checks_pending"]+=["generic_obstacle_semantics","full_masts_and_aboard_occupants","UUID_recording_context"]
        records.append(r)
    p=ROOT/"staging/candidates/mcad_rgb.jsonl";new="".join(json.dumps(r,ensure_ascii=False)+"\n"for r in records)
    if p.exists()and p.read_text(encoding="utf8")!=new:raise ValueError("Existing candidate manifest differs")
    p.write_text(new,encoding="utf8")
    report={"state":"ORIGINAL_RGB_TRAIN_ACQUIRED_NOT_APPROVED","publisher_train_images":len(ims),"native_train_labels_parsed":len(inventory),"selected_max6_boxes_geometry_valid":len(selected),"native_geometry_invalid":sum(bool(r["issues"])for r in inventory),"decoded_staged":len(records),"failures":failures,"native_train_positive_images":dict(Counter(b for r in inventory for b in{v["native_class"]for v in r["boxes"]})),"heldout_image_entries_opened":0,"heldout_label_entries_parsed":0,"ir_images_opened":0,"range_note":"Central-directory/block-cache reads may contain raw bytes adjacent to a selected entry; no adjacent held-out entry is extracted, parsed or decoded.","annotation_index_range_bytes":st.bytes_read,"candidate_manifest":str(p),"candidate_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),"release_approvals":0}
    (ROOT/"reports/mcad_rgb_acquisition.json").write_text(json.dumps(report,indent=2)+"\n",encoding="utf8");print(json.dumps(report,indent=2),flush=True)
if __name__=="__main__":main()
