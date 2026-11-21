#!/usr/bin/env python3
"""Fetch selected non-solid RAR4 members over validated HTTP ranges.
Only archive metadata and explicitly selected original payloads are read. The
publisher CRC is checked by bsdtar while extracting into stdout, not its paths.
"""
import argparse,json,hashlib,subprocess,sys,zlib,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'vendor'))
from inspect_remote_zip import HTTPRangeFile,safe_name

def fetch_member(url,size,item,out,main_prefix,tar):
 if item['bytes']>250_000_000 or item['compressed_bytes']>250_000_000:raise ValueError('Selected member exceeds 250 MB per-file budget')
 parts=safe_name(item['name']);dest=out.joinpath(*parts);dest.parent.mkdir(parents=True,exist_ok=True);fragment=dest.with_suffix(dest.suffix+'.member.rar');s=HTTPRangeFile(url,size);s.seek(item['header_offset']);amount=item['header_size']+item['compressed_bytes'];part=fragment.with_suffix('.part')
 body=bytes([0x7b,0,0x40,7,0]);end=(zlib.crc32(body)&0xffff).to_bytes(2,'little')+body
 with part.open('wb') as f:
  f.write(main_prefix);done=0;notice=time.monotonic()
  while done<amount:
   data=s.read(min(4*1024*1024,amount-done))
   if not data:raise ValueError('Truncated archive member')
   f.write(data);done+=len(data)
   if time.monotonic()-notice>15:print(item['name'],done,'/',amount,'bytes',flush=True);notice=time.monotonic()
  f.write(end)
 part.replace(fragment)
 result=subprocess.run([tar,'-xOf',str(fragment),item['name']],capture_output=True)
 if result.returncode:raise ValueError('Archive CRC/decompression failure: '+result.stderr.decode(errors='replace'))
 if len(result.stdout)!=item['bytes']:raise ValueError('Extracted member length differs from native metadata')
 temp=dest.with_suffix(dest.suffix+'.part');temp.write_bytes(result.stdout);temp.replace(dest)
 return {'source_url':url,'source_size':size,'member':item['name'],'source_header_offset':item['header_offset'],'source_data_offset':item['data_offset'],'compressed_bytes':item['compressed_bytes'],'bytes':dest.stat().st_size,'sha256':hashlib.sha256(result.stdout).hexdigest(),'crc_checked_by':'bsdtar','extracted_to':str(dest)}
def main():
 ap=argparse.ArgumentParser();ap.add_argument('--inventory',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);ap.add_argument('--member',action='append',required=True);ap.add_argument('--tar',default='tar');a=ap.parse_args();m=json.loads(a.inventory.read_text());a.out.mkdir(parents=True,exist_ok=True)
 s=HTTPRangeFile(m['url'],m['archive_bytes']);prefix=s.read(20)
 # SMD's non-solid RAR4 main header occupies exactly 13 bytes after its magic.
 if prefix[:7]!=b'Rar!\x1a\x07\x00' or prefix[9]!=0x73 or int.from_bytes(prefix[12:14],'little')!=13:raise ValueError('Unsupported RAR main header')
 if int.from_bytes(prefix[10:12],'little')&0x8:raise ValueError('Solid RAR archives are not supported')
 rows=[]
 for name in a.member:
  matches=[x for x in m['members'] if x['name']==name]
  if len(matches)!=1:raise ValueError('Member missing or ambiguous')
  item=matches[0]
  if item['flags']&0x10:raise ValueError('Solid member cannot be decoded independently')
  row=fetch_member(m['url'],m['archive_bytes'],item,a.out,prefix,a.tar);rows.append(row);(a.out/'selected_member_receipts.json').write_text(json.dumps(rows,indent=2)+'\n');print(json.dumps(row),flush=True)
if __name__=='__main__':main()
