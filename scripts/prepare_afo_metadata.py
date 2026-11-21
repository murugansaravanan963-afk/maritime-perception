"""Inventory publisher AFO train labels, keeping test image/label payloads sealed."""
import hashlib,json,math,zipfile
from collections import Counter
from pathlib import Path
from inspect_remote_zip import HTTPRangeFile,safe_name
ROOT=Path(__file__).resolve().parents[1]
def save_same(path,data):
    path.parent.mkdir(parents=True,exist_ok=True)
    if path.exists():
        if path.read_bytes()!=data:raise ValueError("Pinned original changed: "+str(path))
    else:path.write_bytes(data)
def main():
    research=ROOT/"research/expansion_20261024"
    ix=json.loads((research/"afo_archive_index.json").read_text())
    base=ROOT/"raw/afo";base.mkdir(exist_ok=True)
    stream=HTTPRangeFile(ix["resolved_archive_url"],ix["archive_bytes"])
    root="PART_1/PART_1/";issues=[];anns=[];missing=[]
    with zipfile.ZipFile(stream) as z:
        names=set(z.namelist());splits={}
        for f in sorted(names):
            if f.startswith(root) and f[len(root):].count("/")==0 and f.endswith(".txt"):
                data=z.read(f);save_same(base/"metadata"/Path(f).name,data)
                if Path(f).name!="readme.txt":splits[Path(f).stem]=[s.strip() for s in data.decode("utf-8-sig").splitlines() if s.strip()]
        for a in splits:
            for b in splits:
                if a!=b and set(splits[a])&set(splits[b]):raise ValueError("Publisher split overlap")
        classdata=z.read(root+"6categories.names")
        save_same(base/"metadata/6categories.names",classdata)
        classes=classdata.decode("utf-8-sig").splitlines()
        if classes!=["human","wind/sup-board","boat","bouy","sailboat","kayak"]:raise ValueError("Native class order changed")
        for name in sorted(splits["train"]):
            f=root+"6categories/"+Path(name).with_suffix(".txt").name
            if f not in names:missing.append(name);continue
            data=z.read(f);dest=base/"annotations_train"/Path(f).name;save_same(dest,data);boxes=[]
            for num,line in enumerate(data.decode("utf-8-sig").splitlines(),1):
                parts=line.split()
                if not parts:continue
                if len(parts)!=5:issues.append([name,num,"wrong_field_count"]);continue
                c=int(parts[0]);x,y,w,h=map(float,parts[1:])
                if c not in range(6) or not all(math.isfinite(v) for v in [x,y,w,h]) or not(w>0 and h>0 and x-w/2>=0 and x+w/2<=1 and y-h/2>=0 and y+h/2<=1):issues.append([name,num,"invalid_native_geometry"])
                boxes.append({"class_id":c,"native_yolo":[x,y,w,h],"native_row":num})
            anns.append({"name":name,"annotation_member":f,"annotation_path":str(dest),"annotation_sha256":hashlib.sha256(data).hexdigest(),"boxes":boxes})
    save_same(research/"afo_train_annotation_inventory.jsonl","".join(json.dumps(a)+"\n" for a in anns).encode())
    report={"source":"AFO publisher version1","publisher_splits":{k:len(v) for k,v in splits.items()},"train_annotation_files_read":len(anns),"test_annotation_payloads_read":0,"missing_train_annotations":missing,"native_geometry_issues":issues,"train_class_image_positives":dict(Counter(c for a in anns for c in {b["class_id"] for b in a["boxes"]})),"native_class_order":classes,"range_bytes_read":stream.bytes_read,"release_approvals":0}
    (research/"afo_train_inventory_summary.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2)[:6000])
if __name__=="__main__":main()
