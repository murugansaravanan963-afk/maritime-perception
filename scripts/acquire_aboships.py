#!/usr/bin/env python3
"""Acquire precisely indexed ABOships PNGs; ZIP CRC + decode, never approvals.

Uses publisher original CSV; its width/height columns are box dimensions, not
image dimensions. Generic Boat/Miscboat/Miscellaneous make the WHOLE frame
unresolved, so this acquisition selects only fully mapped native frames. Visual
completeness, grouping and splits remain separate release requirements.
"""
import argparse,csv,hashlib,json,sys,time,zipfile,zlib
from collections import defaultdict,Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
from inspect_remote_zip import safe_name
from remote_zip_member import fetch_member
from stage_candidates import make_record,check_image
from release_gate import canonical_hash
ROOT=Path(__file__).resolve().parents[1]
MAPPING={'cargoship':0,'cruiseship':0,'ferry':0,'militaryship':0,'motorboat':0,'passengership':0,'sailboat':1,'seamark':3}
def native_rows(path):
    grouped=defaultdict(list)
    with path.open(encoding='utf-8-sig',newline='') as f:
        for n,row in enumerate(csv.DictReader(f),2):
            b=[int(row[k]) for k in ('xmin','ymin','xmax','ymax')]
            if b[2]-b[0]!=int(row['width']) or b[3]-b[1]!=int(row['height']):
                raise ValueError(f'Native box dimension mismatch at CSV line {n}')
            grouped[row['filename']].append({'native_class':row['class'],'xyxy':b,'csv_line':n})
    return grouped

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--limit',type=int,default=64);ap.add_argument('--workers',type=int,default=4);ap.add_argument('--stage-only',action='store_true');ap.add_argument('--local-archive',type=Path)
    a=ap.parse_args();base=ROOT/'raw/aboships';index=json.loads((base/'remote_archive_index.json').read_text(encoding='utf8'))
    csvpath=base/'ABOshipsDataset/Labels/Vesibussi_Labels.csv';ah=hashlib.sha256(csvpath.read_bytes()).hexdigest();rows=native_rows(csvpath)
    entries={Path(x['name']).stem:x for x in index['entries'] if x['name'].endswith('.png')}
    eligible=[k for k,v in rows.items() if all(b['native_class'].lower() in MAPPING for b in v)]
    eligible.sort(key=lambda k:hashlib.sha256(('20261023:'+k).encode()).hexdigest())
    selected=eligible[:a.limit] if a.limit else eligible
    selected.sort(key=lambda k:entries[k]['header_offset'])
    if a.stage_only:selected=[]
    receipts_path=base/'image_receipts.jsonl';receipts={}
    if receipts_path.exists():receipts={x['native_image_id']:x for x in map(json.loads,receipts_path.read_text(encoding='utf8').splitlines())}
    failures=[]
    def fetch(key):
        e=entries[key];p=base.joinpath(*safe_name(e['name']));p.parent.mkdir(parents=True,exist_ok=True)
        if p.exists():
            data=p.read_bytes()
            if len(data)!=e['bytes'] or format(zlib.crc32(data)&0xffffffff,'08x')!=e['crc32']:raise ValueError('Existing PNG CRC mismatch: '+key)
        else:
            error=None
            for attempt in range(4):
                try:
                    
                    if a.local_archive:
                        with zipfile.ZipFile(a.local_archive) as z:data=z.read(e['name'])
                    else:data=fetch_member(index['archive_url'],index['archive_bytes'],e)
                    if len(data)!=e['bytes'] or format(zlib.crc32(data)&0xffffffff,'08x')!=e['crc32']:raise ValueError('ZIP CRC/length mismatch')
                    tmp=p.with_suffix('.png.part');tmp.write_bytes(data);tmp.replace(p);error=None;break
                except Exception as ex:error=ex;time.sleep(min(2**attempt,8))
            if error:raise error
        return {'native_image_id':key,'member':e['name'],'path':str(p),'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'crc32':e['crc32'],'archive_url':index['archive_url'],'publisher_record':4736931}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        jobs={pool.submit(fetch,k):k for k in selected}
        for n,f in enumerate(as_completed(jobs),1):
            k=jobs[f]
            try:
                rec=f.result();receipts[k]=rec
                with receipts_path.open('a',encoding='utf8') as out:out.write(json.dumps(rec)+'\n')
            except Exception as ex:failures.append({'native_image_id':k,'error':str(ex)})
            if n%16==0:print(json.dumps({'processed':n,'selected':len(selected),'failures':len(failures)}),flush=True)
    # Canonicalise resumable acquisition receipts without counting repeated reads.
    receipts_path.write_text(''.join(json.dumps(v)+'\n' for k,v in sorted(receipts.items())),encoding='utf8')
    records=[]
    for k,receipt in sorted(receipts.items()):
        if k not in rows or any(b['native_class'].lower() not in MAPPING for b in rows[k]):raise ValueError('Unmapped image in clean acquisition')
        r=make_record('aboships:'+k,'aboships',ah,Path(receipt['path']),source_split='publisher_unsplit',group='proposal:aboships:date_'+k[:8],view='boat')
        r['source_revision']='zenodo:4736931:v1.0';r['source_terms']={'license':'CC-BY-4.0','record':'https://zenodo.org/records/4736931'}
        r['native_image_id']=k;r['native_capture_prefix']=k.rsplit('_',1)[0]
        r['source_annotation_detail_sha256']=canonical_hash(rows[k]);r['source_archive_member']=receipt['member']
        r['coordinate_contract']='publisher_CSV_direct_xyxy; width_height_are_box_dimensions; no_origin_shift'
        r['boxes']=[{'class_id':MAPPING[b['native_class'].lower()],'xyxy':b['xyxy'],'native_class':b['native_class'],'native_csv_line':b['csv_line']} for b in rows[k]]
        records.append(check_image(r,(1280,720)))
    target=ROOT/'staging/candidates/aboships.jsonl'
    # Existing raw candidate hashes must remain byte-semantically unchanged.
    if target.exists():
        old={r['id']:r for r in map(json.loads,target.read_text(encoding='utf8').splitlines())}
        for r in records:
            if r['id'] in old and canonical_hash(old[r['id']])!=canonical_hash(r):raise ValueError('Existing raw candidate changed: '+r['id'])
    target.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf8')
    report={'selected':len(selected),'eligible_original_frames':len(eligible),'acquired_and_staged':len(records),'invalid_records':sum(bool(r['issues']) for r in records),'failures':failures,'approvals_created':0,'split':'unassigned','publisher_archive_md5_verified':False,'member_crc_and_sha_verified':True}
    (ROOT/'reports/aboships_acquisition.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8');print(json.dumps(report),flush=True)
    if failures:sys.exit(1)
if __name__=='__main__':main()
