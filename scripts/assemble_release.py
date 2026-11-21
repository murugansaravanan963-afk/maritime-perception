#!/usr/bin/env python3
"""Fill a transferred release's image tree from an older release already on disk.

A patch carries the new release's metadata, labels and only the images the older
release lacks. Every other image is hard-linked (or copied) from the older release
by its content-addressed stem. Run preflight_release.py afterwards; it re-hashes
every image and label.

  python scripts/assemble_release.py --release releases/maritime6_pilot_r2 --from releases/maritime6_pilot_r0 releases/maritime6_pilot_r1
"""
import argparse,json,os,shutil
from pathlib import Path
from release_gate import digest,read_jsonl

def main():
 p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--from',dest='src',type=Path,nargs='+',required=True);a=p.parse_args()
 pool={f.name:f for src in a.src for f in (src/'images').rglob('*') if f.is_file()}
 linked=present=0;missing=[]
 for r in read_jsonl(a.release/'manifest.jsonl'):
  im=r['image'];stem=digest(r['id'].encode())[:32];suffix=(im.get('export') or {}).get('suffix') or Path(im['path']).suffix.lower()
  dest=a.release/'images'/r['split']/(stem+suffix)
  if dest.exists():present+=1;continue
  src=pool.get(dest.name)
  if src is None:missing.append(r['id']);continue
  dest.parent.mkdir(parents=True,exist_ok=True)
  try:os.link(src,dest)
  except OSError:shutil.copy2(src,dest)
  linked+=1
 print(json.dumps({'already_present':present,'linked_from_older_release':linked,'missing':len(missing),'missing_examples':missing[:5]}))
 if missing:raise SystemExit(2)
if __name__=='__main__':main()
