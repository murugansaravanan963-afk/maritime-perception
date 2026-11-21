"""Stage publisher TRAIN-only SeaShips VOC candidates; no annotation/release approvals."""
from pathlib import Path
import json,hashlib,zipfile,xml.etree.ElementTree as ET,collections
from concurrent.futures import ThreadPoolExecutor
from stage_candidates import ROOT,make_record,check_image

EXPECTED_SHA='ebaec0a4de849a3cdb5729e7e761bcbb6329997bd06a003fee86f34145a0bbe2'
MOTOR_TYPES={'ore carrier','bulk cargo carrier','general cargo ship','container ship','fishing boat','passenger ship'}

def main():
 archive=ROOT/'raw/seaships7000/SeaShips(7000).zip'
 audit=json.loads((ROOT/'reports/seaships7000_archive.json').read_text(encoding='utf8'))
 if audit['sha256']!=EXPECTED_SHA or not audit['zip_crc_passed'] or archive.stat().st_size!=audit['bytes']:raise ValueError('Transport pin mismatch')
 manifest=ROOT/'staging/candidates/seaships7000.jsonl';dest=ROOT/'raw/seaships7000/publisher_train'
 if manifest.exists() or dest.exists():raise FileExistsError('Immutable acquisition already exists')
 basis_path=ROOT/'research/expansion_20261025_seaships/source_basis.json'
 basis=json.loads(basis_path.read_text(encoding='utf8'))
 records=[];counts=collections.Counter();extrema=collections.Counter()
 with zipfile.ZipFile(archive) as z:
  # Membership lists only for heldouts. No heldout image/annotation content parsed.
  splits={s:set(z.read('ImageSets/Main/'+s+'.txt').decode('utf8').split()) for s in ['train','val','test','trainval']}
  if len(splits['train'])!=1750 or any(splits[a]&splits[b] for a,b in [('train','val'),('train','test'),('val','test')]):raise ValueError('Unexpected or overlapping publisher splits')
  if splits['trainval']!=splits['train']|splits['val']:raise ValueError('Membership mismatch')
  dest.mkdir()
  (dest/'Annotations').mkdir();(dest/'JPEGImages').mkdir()
  for sid in sorted(splits['train']):
   if not sid.isdecimal():raise ValueError('Unexpected sample id')
   member='Annotations/'+sid+'.xml';ab=z.read(member);e=ET.fromstring(ab);w=int(e.findtext('size/width'));h=int(e.findtext('size/height'))
   if e.findtext('filename')!=sid+'.jpg' or (w,h)!=(1920,1080):raise ValueError('XML/image pairing mismatch')
   ip=dest/'JPEGImages'/(sid+'.jpg');ap=dest/'Annotations'/(sid+'.xml')
   ip.write_bytes(z.read('JPEGImages/'+sid+'.jpg'));ap.write_bytes(ab)
   r=make_record('seaships7000:'+sid,'seaships7000_original',hashlib.sha256(ab).hexdigest(),ip,source_split='train',group='campaign:seaships7000:coastline_surveillance_collection',view='shore')
   r['source_revision']='transport_archive_sha256:'+EXPECTED_SHA
   r['source_archive']=str(archive);r['source_annotation_member']=member;r['source_image_member']='JPEGImages/'+sid+'.jpg'
   r['native_label_path']=str(ap);r['native_annotations']=[]
   r['coordinate_contract']='Pascal_VOC_one_based_inclusive_integer_xyxy_to_zero_based_continuous_exclusive_max: [xmin-1,ymin-1,xmax,ymax]; no clipping'
   r['viewpoint_basis']={'publisher_repository':'https://github.com/jiaming-wang/SeaShips','publisher_revision':basis['official_revision'],'reason':'Official publisher identifies deployed coastline surveillance cameras; every frame still requires physical six-class full-frame inspection.'}
   r['capture_group_basis']='Whole coastline acquisition remains one protective campaign. Numeric filenames and publisher random splits do not certify independent camera sites/recordings.'
   r['source_terms']={'publisher':'https://github.com/jiaming-wang/SeaShips','evidence_path':str(basis_path),'license':'Not specified in official README; no CC or commercial/public redistribution license asserted','use_scope':'User-private research only; official public dataset purpose is ship-detector training/evaluation. Mirror transport provenance remains explicit.','transport_sha256':EXPECTED_SHA,'author_archive_hash_equivalence':'Not established'}
   for o in e.findall('object'):
    name=o.findtext('name');xy=[int(o.findtext('bndbox/'+k)) for k in ['xmin','ymin','xmax','ymax']]
    if not(1<=xy[0]<=xy[2]<=w and 1<=xy[1]<=xy[3]<=h):raise ValueError(('Invalid native VOC',sid,xy))
    extrema['xmin_one']+=xy[0]==1;extrema['ymin_one']+=xy[1]==1;extrema['xmax_width']+=xy[2]==w;extrema['ymax_height']+=xy[3]==h
    counts[name]+=1
    r['native_annotations'].append({'name':name,'xyxy_one_based_inclusive':xy,'truncated':o.findtext('truncated'),'difficult':o.findtext('difficult')})
    if name not in MOTOR_TYPES:
     r['unresolved_objects'].append({'native_class':name,'reason':'Unknown native ship type; physical class unresolved'})
    else:
     r['boxes'].append({'class_id':0,'xyxy':[xy[0]-1,xy[1]-1,xy[2],xy[3]],'native_class':name,'mapping_basis':'Publisher commercial motor-vessel type; candidate only, requires physical full-frame inspection'})
    if o.findtext('difficult')=='1':r['issues'].append('native_difficult_requires_review')
   if not r['boxes']:r['issues'].append('no_mapped_targets')
   if r['unresolved_objects']:r['issues'].append('unresolved_native_classes')
   r['expected_dimensions']=[w,h];records.append(r)
 def decode(r):
  expected=r.pop('expected_dimensions');return check_image(r,tuple(expected))
 with ThreadPoolExecutor(max_workers=8) as pool:records=list(pool.map(decode,records))
 if len({r['id'] for r in records})!=1750:raise ValueError('Duplicate sample IDs')
 manifest.write_text(''.join(json.dumps(r)+'\n' for r in records),encoding='utf8')
 report={'state':'PUBLISHER_TRAIN_CANDIDATES_ONLY_NOT_RELEASED','records':len(records),'native_train_class_boxes':dict(counts),'coordinate_border_evidence':dict(extrema),'decode_failures':sum(not r['image'].get('pixel_sha256') for r in records),'distinct_pixels':len({r['image'].get('pixel_sha256') for r in records}),'structurally_unblocked':sum(not r['issues'] and not r['unresolved_objects'] for r in records),'manifest_sha256':hashlib.sha256(manifest.read_bytes()).hexdigest(),'transport_archive_sha256':EXPECTED_SHA,'heldout_membership_counts':{k:len(v) for k,v in splits.items() if k!='trainval'},'heldout_images_or_annotations_parsed':0,'source_use_basis':basis,'automatic_annotation_approvals':0,'training_released':False}
 (ROOT/'reports/seaships7000_acquisition.json').write_text(json.dumps(report,indent=2),encoding='utf8')
 print(json.dumps({k:v for k,v in report.items() if k!='source_use_basis'},indent=2),flush=True)
if __name__=='__main__':main()
