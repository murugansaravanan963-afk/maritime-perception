"""Stage the author's CC-BY-4.0 Yangtze images without inventing detection labels."""
from pathlib import Path
import json,zipfile,hashlib,collections
from stage_candidates import ROOT,make_record,check_image
from inspect_remote_zip import safe_name
from release_gate import canonical_hash

def main():
    archive=ROOT/'raw/yangtze983/983_new-ship-images.zip'
    meta_path=ROOT/'research/expansion_20261025_vessels/983_author_record.json'
    meta=json.loads(meta_path.read_text(encoding='utf8'));native=meta['files'][0]
    if archive.stat().st_size!=native['size']:raise ValueError('Archive size mismatch')
    h=hashlib.md5();sh=hashlib.sha256()
    with archive.open('rb') as f:
        for b in iter(lambda:f.read(2**20),b''):h.update(b);sh.update(b)
    if 'md5:'+h.hexdigest()!=native['checksum']:raise ValueError('Published archive checksum mismatch')
    manifest=ROOT/'staging/candidates/yangtze983.jsonl'
    if manifest.exists():raise FileExistsError(manifest)
    destination=ROOT/'raw/yangtze983/original'
    if destination.exists():raise FileExistsError(destination)
    destination.mkdir()
    metadata_hash=hashlib.sha256(meta_path.read_bytes()).hexdigest()
    records=[]
    with zipfile.ZipFile(archive) as z:
        files=[i for i in z.infolist() if not i.is_dir()]
        if len(files)!=983 or any(Path(i.filename).suffix.lower()!='.jpg' for i in files):raise ValueError('Author image archive layout changed')
        for i in sorted(files,key=lambda x:x.filename):
            safe_name(i.filename);data=z.read(i)
            path=destination/Path(i.filename).name
            if path.exists():raise ValueError('Name collision')
            path.write_bytes(data)
            r=make_record('yangtze983:'+path.stem,'yangtze983_original',canonical_hash({'native_detection_labels':None,'publisher_record_sha256':metadata_hash}),path,source_split='unspecified',group='campaign:yangtze983:author_river_collection',view='unknown')
            r['source_revision']='publisher_archive_sha256:'+sh.hexdigest()
            r['source_archive']=str(archive);r['source_image_member']=i.filename
            r['publisher_record_sha256']=metadata_hash
            r['source_terms']={'license':'CC-BY-4.0','publisher':'https://zenodo.org/records/14274160','creator':'Ren, Yongmei','evidence_path':str(meta_path),'doi':'10.5281/zenodo.14274160'}
            r['capture_group_basis']='Author reports three Yangtze bridge sites; exact per-frame site/recording identity unavailable. Whole collection protected until verified; filenames not treated as independent events.'
            r['coordinate_contract']='No original detection labels in author image archive; explicit full-frame annotation required'
            r['unresolved_objects']=[{'native_class':'image_collection_ship_context','reason':'Image collection description is not a six-class object label or bounding box. Every supplied frame requires complete visual annotation.'}]
            r['issues']=['no_native_detection_annotations']
            r['checks_pending']+=['complete_manual_detection_annotations','capture_site_and_recording_identity','viewpoint_confirmation']
            records.append(check_image(r))
            if len(records)%200==0:print('Decoded '+str(len(records))+'/983',flush=True)
    manifest.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf8')
    report={'state':'ORIGINAL_AUTHOR_IMAGES_STAGED_NO_DETECTION_LABELS_APPROVED','records':len(records),'decode_failures':sum(not r['image'].get('pixel_sha256') for r in records),'distinct_decoded_pixels':len({r['image'].get('pixel_sha256') for r in records}),'archive_sha256':sh.hexdigest(),'published_md5_verified':True,'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'native_detection_annotation_files':0,'author_license':meta['metadata']['license'],'automatic_label_approvals':0}
    p=ROOT/'reports/yangtze983_acquisition.json';p.write_text(json.dumps(report,indent=2),encoding='utf8');print(json.dumps(report,indent=2),flush=True)
if __name__=='__main__':main()
