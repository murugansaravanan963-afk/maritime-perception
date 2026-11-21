"""Resumable single-stream original archive acquisition with publisher MD5.
No extraction until byte count and checksum match the pinned publisher record.
"""
import json,urllib.request,urllib.error,time,hashlib,sys
from pathlib import Path
root=Path(__file__).resolve().parents[1];source=sys.argv[1]
api=json.loads((root/f'research/{source}_zenodo.json').read_text(encoding='utf8'))
name=sys.argv[2];f=next(x for x in api['files'] if x['key']==name);url=f['links']['self'];size=f['size'];out=root/'raw'/source/name;tmp=out.with_suffix(out.suffix+'.part');out.parent.mkdir(parents=True,exist_ok=True)
if out.exists():tmp=out
for attempt in range(12):
 offset=tmp.stat().st_size if tmp.exists() else 0
 if offset==size:break
 if offset>size:raise ValueError('Oversized original archive partial')
 headers={'User-Agent':'MaritimeDataset/1.0'}
 if offset:headers['Range']=f'bytes={offset}-'
 try:
  with urllib.request.urlopen(urllib.request.Request(url,headers=headers),timeout=60) as resp:
   if offset and (resp.status!=206 or resp.headers.get('Content-Range')!=f'bytes {offset}-{size-1}/{size}'):raise ValueError('Publisher did not honor archive resume range')
   if not offset and resp.status!=200:raise ValueError('Unexpected original archive response')
   mode='ab' if offset else 'wb';last=offset
   with tmp.open(mode) as dst:
    while True:
     block=resp.read(4*1048576)
     if not block:break
     dst.write(block);offset+=len(block)
     if offset>size:raise ValueError('Archive response exceeds publisher length')
     if offset-last>=128*1048576:print(json.dumps({'downloaded_bytes':offset,'archive_bytes':size}),flush=True);last=offset
 except urllib.error.HTTPError as ex:
  if ex.code not in (429,500,502,503,504):raise
  delay=max(30,min(int(ex.headers.get('Retry-After','90')) if ex.headers.get('Retry-After','90').isdigit() else 90,300))
  print(json.dumps({'publisher_status':ex.code,'backoff_seconds':delay,'partial_bytes':offset}),flush=True)
  for _ in range((delay+29)//30):time.sleep(30)
 except (TimeoutError,ConnectionError,OSError) as ex:
  print(json.dumps({'recoverable_error':type(ex).__name__,'partial_bytes':tmp.stat().st_size if tmp.exists() else 0}),flush=True);time.sleep(15)
else:raise RuntimeError('Download retries exhausted; original partial retained')
if not tmp.exists() or tmp.stat().st_size!=size:raise ValueError('Original archive incomplete')
md5=hashlib.md5();sha=hashlib.sha256()
with tmp.open('rb') as src:
 for block in iter(lambda:src.read(8*1048576),b''):md5.update(block);sha.update(block)
if f['checksum']!='md5:'+md5.hexdigest():raise ValueError('Publisher archive MD5 mismatch; no extraction')
if tmp!=out:tmp.replace(out)
receipt={'publisher_record':api['id'],'url':url,'bytes':size,'publisher_md5_verified':True,'md5':md5.hexdigest(),'sha256':sha.hexdigest(),'path':str(out)}
(out.parent/(name+'.receipt.json')).write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8');print(json.dumps(receipt),flush=True)
