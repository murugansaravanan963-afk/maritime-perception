#!/usr/bin/env python3
"""Convert original TRAIN sources to inspectable, unreleased six-class candidates.

This creates no approvals, splits, dataset.yaml or training output. Unknown native
objects remain explicit blockers. Original sources are never edited.
"""
import argparse, json, re, hashlib, zipfile
from collections import Counter, defaultdict
from pathlib import Path
import numpy as np
from PIL import Image
from audit_candidates import image_info, sha, box_issues, write_json
from release_gate import NAMES, canonical_hash

ROOT=Path(__file__).resolve().parents[1]
OLD=Path(r'D:\Projects\maritime-perception')
MAP_LARS={12:2,13:2,14:3,15:4,17:5}
MAP_SEACLIPS={2:0,3:3,5:2,6:1,7:0}
MAP_SDS={1:4,3:0,4:5,5:3}

def jsonl(path):
 with path.open(encoding='utf-8-sig') as f:
  for line in f:
   if line.strip(): yield json.loads(line)

def native_box(xywh):
 x,y,w,h=xywh
 return [x,y,x+w,y+h]

def panoptic_ids(rgb):
 """COCO panoptic RGB -> uint32 ID, preserving individual instances."""
 a=np.asarray(rgb,dtype=np.uint32)
 if a.ndim!=3 or a.shape[2]!=3: raise ValueError('panoptic_mask_not_RGB')
 return a[:,:,0]+256*a[:,:,1]+65536*a[:,:,2]

def mask_geometry(ids,segments):
 """Check mask IDs, pixel area and exact visible-instance boxes, no box repair."""
 found,counts=np.unique(ids,return_counts=True);areas=dict(zip(map(int,found),map(int,counts)))
 declared={s['id'] for s in segments};issues=[];boxes={}
 if len(declared)!=len(segments):issues.append('duplicate_segment_id')
 if set(areas)-declared-{0}:issues.append('mask_has_undeclared_segments')
 h,w=ids.shape
 for s in segments:
  sid=s['id'];x,y,bw,bh=s['bbox'];x1,y1,x2,y2=map(int,[x,y,x+bw,y+bh])
  if any(float(v)!=int(v) for v in s['bbox']) or not(0<=x1<x2<=w and 0<=y1<y2<=h):
   issues.append(f'invalid_native_mask_box:{sid}');continue
  if areas.get(sid,0)!=s['area']:issues.append(f'mask_area_mismatch:{sid}')
  ys,xs=np.nonzero(ids[y1:y2,x1:x2]==sid)
  if not len(xs):issues.append(f'segment_absent_in_box:{sid}');continue
  xy=[int(x1+xs.min()),int(y1+ys.min()),int(x1+xs.max()+1),int(y1+ys.max()+1)]
  # The crop must contain ALL pixels of the instance; never silently truncate.
  if len(xs)!=areas.get(sid,0):issues.append(f'instance_outside_declared_box:{sid}')
  if xy!=[x1,y1,x2,y2]:issues.append(f'mask_box_not_tight:{sid}')
  boxes[sid]=xy
 return boxes,issues

def make_record(rid,source,annhash,path,source_split='train',group=None,view='unknown'):
 return {'id':rid,'source':source,'source_revision':'original_local_train:'+annhash,
  'source_annotation_sha256':annhash,'source_split':source_split,'split':'unassigned',
  'viewpoint':view,'capture_group':group,'independent_unit':rid,'duplicate_cluster':rid,
  'image':{'path':str(path)},'boxes':[],'unresolved_objects':[],
  'issues':[], 'review_state':'PENDING_FULL_FRAME_REVIEW',
  'checks_pending':['full_frame_class_and_box_review','source_terms','recording_group_verification','near_duplicate_review','independent_split_assignment']}

def check_image(r,expected=None):
 p=Path(r['image']['path'])
 try:
  info=image_info(p.read_bytes());r['image'].update(width=info['width'],height=info['height'],sha256=info['image_sha256'],pixel_sha256=info['pixel_sha256'],dhash64=info['dhash64'])
  if expected and (info['width'],info['height'])!=expected:r['issues'].append('source_image_dimension_mismatch')
  r['issues']+=box_issues([{'native_class':b['class_id'],'xyxy':b['xyxy']} for b in r['boxes']],info['width'],info['height'])
 except Exception as e:r['issues'].append('image_failure:'+type(e).__name__+':'+str(e))
 return r

def lars(old):
 ar=old/'data/lars_v1.0.0_annotations/train';p=ar/'panoptic_annotations.json';ab=p.read_bytes();d=json.loads(ab);ah=sha(ab)
 images={i['id']:i for i in d['images']};cats={c['id']:c for c in d['categories']}
 for a in d['annotations']:
  ss=[s for s in a['segments_info'] if cats[s['category_id']]['isthing']]
  if not ss:continue
  im=images[a['image_id']];fn=im['file_name'];prefix=fn.rsplit('_',1)[0]
  r=make_record('lars:'+Path(fn).stem,'lars_v1.0.0',ah,old/'data/lars_images/train/images'/fn,group='proposal:lars:'+prefix,view='boat')
  r['native_image_id']=im['id'];r['native_mask_path']=str(ar/'panoptic_masks'/a['file_name']);r['coordinate_contract']='COCO_xywh_and_RGB_instance_mask_exclusive_max'
  r['source_annotation_detail_sha256']=canonical_hash(a)
  try:
   with Image.open(r['native_mask_path']) as m:
    ids=panoptic_ids(m.convert('RGB'))
    if m.size!=(im['width'],im['height']):r['issues'].append('mask_dimension_mismatch')
   r['source_mask_sha256']=sha(Path(r['native_mask_path']).read_bytes())
   mb,mi=mask_geometry(ids,a['segments_info']);r['issues']+=mi
  except Exception as e:mb={};r['issues'].append('mask_failure:'+str(e))
  for s in ss:
   c=s['category_id'];b={'native_class':cats[c]['name'],'native_category_id':c,'native_segment_id':s['id'],'native_xyxy':native_box(s['bbox'])}
   if c in MAP_LARS:
    r['boxes'].append({'class_id':MAP_LARS[c],'xyxy':native_box(s['bbox']),'native_segment_id':s['id']})
   else:r['unresolved_objects'].append(b)
   if s.get('iscrowd'):r['issues'].append('crowd_instance_requires_review')
  if r['unresolved_objects']:r['issues'].append('unresolved_native_classes')
  r['review_priority']='surface_swimmer_no_ambiguous_native_classes' if any(b['class_id']==4 for b in r['boxes']) and not r['unresolved_objects'] else 'normal'
  yield check_image(r,(im['width'],im['height']))

def coco(old,source,step):
 rel='SeaClips/seaclips-train.json' if source=='seaclips' else 'SeaDronesSee_v4/annotations/instances_train.json'
 p=old/'data'/rel
 if source=='seadronessee' and (ROOT/'raw/seadronessee_aux/selected_train.json').exists():p=ROOT/'raw/seadronessee_aux/selected_train.json'
 ab=p.read_bytes();d=json.loads(ab);ah=sha((ROOT/'raw/seadronessee_aux/instances_train_original.json').read_bytes()) if source=='seadronessee' and p.parent.name=='seadronessee_aux' else sha(ab);cats={x['id']:x['name'] for x in d['categories']};by=defaultdict(list)
 for a in d['annotations']:by[a['image_id']].append(a)
 mapping=MAP_SEACLIPS if source=='seaclips' else MAP_SDS
 ims=sorted(d['images'],key=lambda i:(str(i.get('video_id',i.get('source',{}).get('video',''))),i.get('frame_id',i.get('source',{}).get('frame_no',0)),i['id']))
 for im in ims:
  aa=by[im['id']]
  if not aa:continue
  frame=im.get('frame_id',im.get('source',{}).get('frame_no',0))
  # This is a review-budget sampling interval, NOT proof of independence.
  if frame%step:continue
  if source=='seaclips':path=old/'data/SeaClips/train'/im['file_name'];group='proposal:seaclips:video_'+str(im['video_id']);view='boat'
  else:path=(ROOT/'raw/seadronessee_aux/images/train' if p.parent.name=='seadronessee_aux' else old/'data/SeaDronesSee_v4/images/train')/im['file_name'];group='proposal:seadronessee:'+im.get('source',{}).get('video','unknown');view='aerial'
  r=make_record(source+':'+str(im['id']),source,ah,path,group=group,view=view)
  r['native_image_id']=im['id'];r['native_image_metadata']=im;r['source_annotation_detail_sha256']=canonical_hash(aa);r['coordinate_contract']='native_COCO_xywh_exclusive_max';r['frame_sampling_step']=step
  for a in aa:
   c=a['category_id'];xy=native_box(a['bbox'])
   if c in mapping:r['boxes'].append({'class_id':mapping[c],'xyxy':xy,'native_annotation_id':a.get('id')})
   else:r['unresolved_objects'].append({'native_class':cats[c],'native_xyxy':xy,'native_annotation_id':a.get('id')})
   if a.get('iscrowd'):r['issues'].append('crowd_instance_requires_review')
  if r['unresolved_objects']:r['issues'].append('unresolved_native_classes')
  # Preserve missing images as failures rather than guessing another file path.
  yield check_image(r,(im['width'],im['height']))

def openimages():
 p=ROOT/'reports/openimages_canoe_photo_manifest.jsonl'
 for a in jsonl(p):
  ah=canonical_hash(a['native_annotations']);prov=a['provenance'];group='proposal:openimages:author:'+sha(prov.get('AuthorProfileURL','unknown').encode())[:24]
  r=make_record(a['id'],'openimages_v7',ah,Path(a['image_path']),group=group)
  r['coordinate_contract']='native_OpenImages_normalized_xyxy';r['source_attribution']=prov;r['native_annotations']=a['native_annotations']
  r['checks_pending']+=['on_water_vs_stored_canoe','viewpoint_review','native_extent_convention']
  for b in a['native_annotations']:
   xy=[float(b[k])*a['width' if k.startswith('X') else 'height'] for k in ['XMin','YMin','XMax','YMax']]
   if b['LabelName']=='/m/0ph39':r['boxes'].append({'class_id':2,'xyxy':xy,'native_class':'Canoe','native_flags':{k:b[k] for k in ['IsDepiction','IsGroupOf','IsOccluded','IsTruncated']}})
   else:r['unresolved_objects'].append({'native_label':b['LabelName'],'native_xyxy':xy})
  # Native people, boat and animal boxes are not discarded as background.
  if r['unresolved_objects']:r['issues'].append('cooccurring_native_labels_require_context')
  yield check_image(r,(a['width'],a['height']))

def openimages_people():
 audit=json.loads((ROOT/'reports/openimages_people_photo_audit.json').read_text(encoding='utf8'))
 if audit['failures'] or audit['downloaded_decoded']!=audit['selected_images']:raise ValueError('People context acquisition incomplete')
 existing={r['id']:r for r in jsonl(ROOT/'staging/candidates/openimages.jsonl')};overlap=[]
 for a in jsonl(ROOT/'reports/openimages_people_photo_manifest.jsonl'):
  if a['id'] in existing:
   if existing[a['id']]['image']['sha256']!=a.get('image_sha256'):raise ValueError('Overlapping Open Images payload differs')
   if canonical_hash(existing[a['id']]['native_annotations'])!=canonical_hash(a['native_annotations']):raise ValueError('Overlapping native annotations differ')
   overlap.append(a['id']);continue
  ah=canonical_hash(a['native_annotations']);prov=a['provenance'];group='proposal:openimages:author:'+sha(prov.get('AuthorProfileURL','unknown').encode())[:24]
  r=make_record(a['id'],'openimages_v7',ah,Path(a['image_path']),group=group)
  r['source_attribution']=prov;r['native_annotations']=a['native_annotations'];r['coordinate_contract']='native_OpenImages_normalized_xyxy'
  r['issues']+=a['issues'];r['issues'].append('person_in_water_status_not_established')
  r['checks_pending']+=['in_water_vs_shore_aboard_or_pool','person_visible_body_extent','viewpoint_review']
  r['unresolved_objects']=[{'native_label':b['LabelName'],'native_xyxy':[float(b[k])*a['width' if k.startswith('X') else 'height'] for k in ['XMin','YMin','XMax','YMax']]} for b in a['native_annotations']]
  # Garments/caps/generic people are NEVER translated automatically to swimmers.
  yield check_image(r,(a['width'],a['height']))
 write_json(ROOT/'reports/openimages_people_canoe_overlap.json',{'shared_original_image_ids':overlap,'count':len(overlap),'policy':'One canonical Open Images record per image; all supplied native labels already retained by canoe inventory. No double counting.'})

def debris(source):
 name='debris' if source=='dsix' else source
 for a in jsonl(ROOT/f'reports/{name}_native_manifest.jsonl'):
  if a['source']!=source:continue
  target=ROOT/'staging/materialized'/source/(sha(a['id'].encode())[:32]+Path(a['image_member']).suffix.lower())
  if not target.exists():
   target.parent.mkdir(parents=True,exist_ok=True)
   with zipfile.ZipFile(a['archive']) as z:data=z.read(a['image_member'])
   if sha(data)!=a['image_sha256']:raise ValueError('Native archived image changed: '+a['id'])
   target.write_bytes(data)
  group='proposal:'+source+':'+a.get('group_proposal','unknown')
  r=make_record(a['id'],source,a['annotation_sha256'],target,a['source_split'],group,view='shore' if source!='dsix' else 'boat')
  r['source_archive']=a['archive'];r['source_image_member']=a['image_member'];r['source_annotation_member']=a['annotation_member'];r['coordinate_contract']=a['coordinate_contract'];r['issues']+=a['issues']
  r['boxes']=[{'class_id':5,'xyxy':b['xyxy'],'native_class':b['native_class']} for b in a['boxes']]
  r['checks_pending']+=['floating_vs_onshore','box_extent_convention']
  yield check_image(r,(a['width'],a['height']))

def stage(source,old,step):
 it=lars(old) if source=='lars' else coco(old,source,step) if source in ['seaclips','seadronessee'] else openimages() if source=='openimages' else openimages_people() if source=='openimages_people' else debris(source)
 out=ROOT/'staging/candidates';out.mkdir(parents=True,exist_ok=True);dest=out/(source+'.jsonl');tmp=dest.with_suffix('.jsonl.partial');rows=[]
 with tmp.open('w',encoding='utf8') as f:
  for n,r in enumerate(it,1):
   r['issues']=sorted(set(r['issues']));f.write(json.dumps(r,ensure_ascii=False)+'\n');rows.append(r)
   if n%100==0:print(source,n,'staged unreleased candidates',flush=True)
 tmp.replace(dest)
 counts=Counter();primary=Counter();groups=defaultdict(set)
 for r in rows:
  if any(x.startswith('image_failure') for x in r['issues']):continue
  for c in {b['class_id'] for b in r['boxes']}:
   counts[NAMES[c]]+=1;groups[NAMES[c]].add(r['capture_group'])
   if r['viewpoint'] in ['boat','shore','surface']:primary[NAMES[c]]+=1
 report={'source':source,'state':'CANDIDATE_ONLY_NOT_RELEASED','records':len(rows),'mapped_candidate_images_per_class':dict(counts),'mapped_primary_candidate_images_per_class':dict(primary),'group_proposals_per_class':{k:len(v) for k,v in groups.items()},'structural_issue_records':sum(bool(r['issues']) for r in rows),'image_failures':sum(any(x.startswith('image_failure') for x in r['issues']) for r in rows),'fully_reviewed_records':0,'publisher_test_payloads_opened':0,'warning':'Mapped candidates are NOT approved labels. No split has been assigned; ambiguous native objects and pending review are retained.'}
 write_json(ROOT/f'reports/staging_{source}.json',report);print(json.dumps(report,indent=2))
 return report

def main():
 p=argparse.ArgumentParser();p.add_argument('source',choices=['lars','seaclips','seadronessee','openimages','openimages_people','dsix','oostpoort','instream_wood']);p.add_argument('--old',type=Path,default=OLD);p.add_argument('--frame-step',type=int,default=5);a=p.parse_args()
 if a.frame_step<1:raise ValueError('frame-step must be positive')
 stage(a.source,a.old,a.frame_step)
if __name__=='__main__':main()
