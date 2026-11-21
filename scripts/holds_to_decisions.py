#!/usr/bin/env python3
"""Build decisions.txt for an assist batch from holds.txt lines "<page> <slot> <note>".

Every page must be listed in reviewed.txt ("<page>" per line) so frames on pages
that were never viewed cannot silently pass. Unlisted frames on viewed pages pass.

  python scripts/holds_to_decisions.py reports/assist_duth_001 "<pass note>"
"""
import json,sys
from pathlib import Path
batch=Path(sys.argv[1]);note=sys.argv[2];partial='--partial' in sys.argv
index=[json.loads(l) for l in (batch/'index.jsonl').read_text().splitlines()]
key=lambda x:(Path(x['evidence_path']).stem.split('_')[1],x['slot'])
holds={}
if (batch/'holds.txt').exists():
 for l in (batch/'holds.txt').read_text(encoding='utf8').splitlines():
  if l.strip():p,s,n=l.split(' ',2);holds[(p.zfill(4),s)]=n
viewed={l.strip().zfill(4) for l in (batch/'reviewed.txt').read_text().splitlines() if l.strip()}
pages={key(x)[0] for x in index}
if partial:index=[x for x in index if key(x)[0] in viewed]
elif pages-viewed:raise SystemExit('Pages not viewed: '+','.join(sorted(pages-viewed)[:20]))
if set(holds)-{key(x) for x in index}:raise SystemExit('Unknown hold slots: '+str(sorted(set(holds)-{key(x) for x in index})))
(batch/'decisions.txt').write_text(''.join(f"{x['id']}\thold\t{holds[key(x)]}\n" if key(x) in holds else f"{x['id']}\tpass\t{note}\n" for x in index),encoding='utf8')
print(len(index),'frames',len(holds),'holds')
