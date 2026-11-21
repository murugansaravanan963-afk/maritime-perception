#!/usr/bin/env python3
"""Verified official-source acquisition. Raw inputs are never training admissions."""
import argparse,concurrent.futures,hashlib,json,os,sys,time,urllib.request
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]

def download(item):
    out=ROOT/item["path"];out.parent.mkdir(parents=True,exist_ok=True)
    def check(path):
        if not path.exists() or path.stat().st_size!=item["bytes"]:return False
        h=hashlib.md5();s=hashlib.sha256()
        with path.open("rb") as f:
            for chunk in iter(lambda:f.read(4*1024*1024),b""):h.update(chunk);s.update(chunk)
        return (not item.get("md5") or h.hexdigest()==item["md5"]) and (not item.get("sha256") or s.hexdigest()==item["sha256"])
    if check(out):return {"id":item["id"],"state":"cached_verified","path":str(out),"bytes":out.stat().st_size,"sha256":item.get("sha256"),"publisher_md5":item.get("md5"),"url":item["url"]}
    if out.exists():raise ValueError("Existing final file failed verification: "+str(out))
    part=out.with_suffix(out.suffix+".part")
    last=None
    for attempt in range(3):
        try:
            offset=part.stat().st_size if part.exists() else 0
            headers={"User-Agent":"MaritimeDataset/1.0"}
            if offset:headers["Range"]=f"bytes={offset}-"
            req=urllib.request.Request(item["url"],headers=headers)
            with urllib.request.urlopen(req,timeout=35) as resp:
                if "text/html" in resp.headers.get("Content-Type","") and item["path"].endswith((".zip",".json",".rar")):
                    raise ValueError("Expected data file but server returned HTML; access/download needs attention")
                if offset and (resp.status!=206 or not resp.headers.get("Content-Range","").startswith(f"bytes {offset}-")):
                    offset=0
                total=offset;last_notice=time.monotonic()
                with part.open("ab" if offset else "wb") as f:
                    while True:
                        chunk=resp.read(1024*1024)
                        if not chunk:break
                        f.write(chunk);total+=len(chunk)
                        if total>item["bytes"]:raise ValueError("Response exceeds declared file size")
                        if time.monotonic()-last_notice>15:
                            print(f"{item['id']}: {total}/{item['bytes']} bytes",flush=True);last_notice=time.monotonic()
            if not check(part):raise ValueError("Expected byte count or publisher MD5 did not match")
            part.replace(out)
            digest=hashlib.sha256()
            with out.open("rb") as f:
                for chunk in iter(lambda:f.read(4*1024*1024),b""):digest.update(chunk)
            return {"id":item["id"],"state":"download_verified","path":str(out),"bytes":out.stat().st_size,"sha256":digest.hexdigest(),"publisher_md5":item.get("md5"),"url":item["url"]}
        except Exception as exc:
            last=str(exc)
            # A complete corrupt part must never be retried by appending data.
            if part.exists() and part.stat().st_size>=item["bytes"]:
                bad=part.with_suffix(part.suffix+f".rejected{attempt}")
                part.replace(bad)
            print(f"{item['id']}: attempt {attempt+1} failed: {last}",flush=True)
            time.sleep(min(2**attempt,4))
    return {"id":item["id"],"state":"failed","error":last,"url":item["url"]}

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--profile",default="research")
    ap.add_argument("--workers",type=int,default=3)
    a=ap.parse_args()
    manifest=json.loads((ROOT/"configs/downloads.json").read_text())
    items=[x for x in manifest["files"] if a.profile in x["profiles"]]
    if not items:raise SystemExit("No matching profile")
    receipt=ROOT/"reports"/f"acquisition_{a.profile}.json"
    rows=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs=[pool.submit(download,x) for x in items]
        for job in concurrent.futures.as_completed(jobs):
            result=job.result();rows.append(result)
            receipt.write_text(json.dumps({"profile":a.profile,"files":rows},indent=2)+"\n")
            print(json.dumps(result),flush=True)
    if any(x["state"]=="failed" for x in rows):raise SystemExit(1)
if __name__=="__main__":main()

