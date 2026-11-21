#!/usr/bin/env python3
"""Native data inventory. No training admission, remapping, clipping or test payload reads."""
import argparse,hashlib,io,json,math,re,zipfile
from collections import Counter,defaultdict
from pathlib import Path
import xml.etree.ElementTree as ET
from PIL import Image,ImageFile
import numpy as np
ImageFile.LOAD_TRUNCATED_IMAGES=False
ROOT=Path(__file__).resolve().parents[1]
def sha(data):return hashlib.sha256(data).hexdigest()
def write_json(p,v):p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(v,indent=2,ensure_ascii=False)+'\n',encoding='utf8')
def image_info(data):
 with Image.open(io.BytesIO(data)) as im:im.verify()
 with Image.open(io.BytesIO(data)) as im:
  im.load();rgb=im.convert('RGB'); a=np.asarray(rgb.resize((9,8)).convert('L')); bits=(a[:,1:]>a[:,:-1]).flatten();dh=sum(int(v)<<i for i,v in enumerate(bits))
  return {'width':im.width,'height':im.height,'mode':im.mode,'image_sha256':sha(data),'pixel_sha256':sha(str(rgb.size).encode()+rgb.tobytes()),'dhash64':format(dh,'016x')}
def box_issues(boxes,w,h,native=True):
 issues=[];seen=set()
 for i,b in enumerate(boxes):
  c=b['native_class'];xy=b['xyxy'];k=(str(c),*xy)
  if k in seen:issues.append(f'duplicate_box:{i}')
  seen.add(k)
  if not all(math.isfinite(float(v)) for v in xy):issues.append(f'nonfinite_box:{i}');continue
  x1,y1,x2,y2=xy
  if not(0<=x1<x2<=w and 0<=y1<y2<=h):issues.append(f'out_of_bounds_or_degenerate:{i}')
 return issues
def source_group(name):
 # Proposals only. NOT certified independent recordings.
 s=Path(name).stem
 m=re.match(r'(20\d{2}[-_]\d{2}[-_]\d{2})',s)
 return m.group(1) if m else re.sub(r'[-_, ]*\d[\d,]*$','',s) or 'unknown'
def audit_debris():
 records=[];summaries=[]
 for p in sorted((ROOT/'raw/iwhr').glob('*.zip'))+sorted((ROOT/'raw/dsix').glob('*.zip')):
  source='iwhr' if p.parent.name=='iwhr' else 'dsix';current=[]
  with zipfile.ZipFile(p) as z:
   allnames=set(z.namelist())
   names=sorted(n for n in allnames if n.lower().endswith('.jpg') and (source=='iwhr' or '/images/train/' in n))
   for idx,n in enumerate(names):
    r={'id':source+':'+Path(n).stem,'source':source,'archive':str(p),'image_member':n,'source_split':'unspecified' if source=='iwhr' else 'train','state':'unreviewed_candidate','issues':[]}
    try:
     data=z.read(n);r.update(image_info(data))
     if source=='iwhr':
      ann=n.replace('/JPEGImages/','/Annotations/').rsplit('.',1)[0]+'.xml'
      if ann not in allnames:
       matches=[x for x in allnames if x.endswith('/'+Path(n).stem+'.xml')]
       if len(matches)!=1:raise ValueError('annotation_pair_not_unique')
       ann=matches[0]
      ab=z.read(ann);xml=ET.fromstring(ab);r['original_filename']=xml.findtext('filename');r['group_proposal']=source_group(r['original_filename'] or '')
      dims=(int(xml.findtext('size/width')),int(xml.findtext('size/height')))
      if dims!=(r['width'],r['height']):r['issues'].append('xml_image_dimension_mismatch')
      boxes=[{'native_class':o.findtext('name'),'xyxy':[float(o.findtext('bndbox/'+k)) for k in ['xmin','ymin','xmax','ymax']],'difficult':o.findtext('difficult'),'truncated':o.findtext('truncated')} for o in xml.findall('object')]
      if any(b['native_class']!='floater' for b in boxes):r['issues'].append('unexpected_class')
      r['coordinate_contract']='native_xml_xyxy_origin_not_yet_confirmed'
     else:
      ann=n.replace('/images/','/labels/').rsplit('.',1)[0]+'.txt';ab=z.read(ann);boxes=[]
      for line in ab.decode('utf-8-sig').splitlines():
       if not line.strip():continue
       t=line.split()
       if len(t)!=5:raise ValueError('malformed_yolo_line')
       c=float(t[0]);x,y,w,h=map(float,t[1:])
       if c!=int(c) or not 0<=c<6:raise ValueError('unexpected_class')
       boxes.append({'native_class':int(c),'xyxy':[(x-w/2)*r['width'],(y-h/2)*r['height'],(x+w/2)*r['width'],(y+h/2)*r['height']]})
      r['coordinate_contract']='yolo_normalized_center_xywh';r['group_proposal']='unknown'
     r.update(annotation_member=ann,annotation_sha256=sha(ab),boxes=boxes)
     r['issues']+=box_issues(boxes,r['width'],r['height'])
    except Exception as e:r['issues'].append(type(e).__name__+':'+str(e))
    current.append(r);records.append(r)
    if (idx+1)%250==0:print(p.name,idx+1,'images decoded',flush=True)
  fc=Counter();bc=Counter()
  for r in current:
   bc.update(b['native_class'] for b in r.get('boxes',[]));fc.update(set(b['native_class'] for b in r.get('boxes',[])))
  summaries.append({'archive':p.name,'audited_images':len(current),'images_with_issues':sum(bool(r['issues']) for r in current),'native_images_per_class':dict(fc),'native_boxes_per_class':dict(bc),'proposed_group_counts':dict(Counter(r.get('group_proposal','unknown') for r in current))})
 # Hash duplicate discovery across sources. DHash matches are review suggestions only.
 exact=defaultdict(list);pixels=defaultdict(list);near=[]
 for r in records:
  if 'pixel_sha256' in r:exact[r['image_sha256']].append(r['id']);pixels[r['pixel_sha256']].append(r['id'])
 valid=[r for r in records if 'dhash64' in r]
 # Vectorized XOR/popcount avoids millions of Python operations; <=5 catches candidates, not proof.
 lut=np.array([i.bit_count() for i in range(256)],dtype=np.uint8)
 vals=np.array([int(r['dhash64'],16) for r in valid],dtype=np.uint64)
 for i,v in enumerate(vals):
  dist=lut[np.bitwise_xor(v,vals[i+1:]).view(np.uint8).reshape(-1,8)].sum(axis=1)
  for off in np.flatnonzero(dist<=5):
   j=i+1+int(off)
   if valid[i]['pixel_sha256']!=valid[j]['pixel_sha256']:near.append({'a':valid[i]['id'],'b':valid[j]['id'],'dhash_distance':int(dist[off])})
 out=ROOT/'reports';out.mkdir(exist_ok=True)
 with (out/'debris_native_manifest.jsonl').open('w',encoding='utf8') as f:
  for r in records:f.write(json.dumps(r,ensure_ascii=False)+'\n')
 report={'state':'AUDIT_ONLY_NOT_RELEASED','test_payloads_opened':0,'validation_payloads_opened':0,'sources':summaries,'exact_file_duplicate_groups':[v for v in exact.values() if len(v)>1],'decoded_pixel_duplicate_groups':[v for v in pixels.values() if len(v)>1],'near_duplicate_candidates':near,'near_duplicate_method':'64bit dHash Hamming <=5, visual confirmation required','admitted_images':0,'remaining_checks':['full-frame six-class completeness and class meaning','XML coordinate origin convention','recording/site groups and temporal thinning','visually confirm duplicate candidates','class-balanced disjoint split coverage']}
 write_json(out/'debris_native_audit.json',report);print(json.dumps({k:v for k,v in report.items() if k not in ['near_duplicate_candidates','sources']}));print('SUMMARY',json.dumps(summaries))
def audit_waterscenes():
 d=ROOT/'raw/waterscenes'; ids=[Path(x.strip()).stem for x in (d/'train.txt').read_text().splitlines() if x.strip()]; fc=Counter();bc=Counter();issues=[];candidate=Counter();rows=[];hyp=Counter();mapping={0:'out_of_scope',1:'navigation_marker',2:'out_of_scope',3:'unresolved',4:'unresolved',5:'unresolved',6:'small_craft'}
 with zipfile.ZipFile(d/'detection.zip') as z:
  for id in ids:
   ab=z.read('detection/xml/'+id+'.xml');x=ET.fromstring(ab);w=int(x.findtext('size/width'));h=int(x.findtext('size/height'));objects=x.findall('object');names=[o.findtext('name') for o in objects];fc.update(set(names));bc.update(names)
   lines=[list(map(float,l.split())) for l in z.read('detection/yolo/'+id+'.txt').decode().splitlines() if l.strip()]
   if len(lines)!=len(objects):issues.append([id,'xml_yolo_object_count_mismatch']);continue
   unresolved=any(int(t[0]) in [3,4,5] for t in lines)
   if not unresolved:candidate.update(set(mapping[int(t[0])] for t in lines if mapping[int(t[0])] not in ['out_of_scope']))
   for o,t in zip(objects,lines):
    x1,y1,x2,y2=[float(o.findtext('bndbox/'+k)) for k in ['xmin','ymin','xmax','ymax']];target=np.array(t[1:]);direct=np.array([(x1+x2)/2/w,(y1+y2)/2/h,(x2-x1)/w,(y2-y1)/h]);voc=np.array([(x1-1+x2)/2/w,(y1-1+y2)/2/h,(x2-x1+1)/w,(y2-y1+1)/h])
    hyp['direct_match']+=int(np.max(np.abs(target-direct))<0.000002);hyp['voc1_match']+=int(np.max(np.abs(target-voc))<0.000002)
   rows.append({'id':id,'original_filename':x.findtext('filename'),'native_classes':sorted(set(names)),'ambiguous_vessel_frame':unresolved,'annotation_sha256':sha(ab)})
 with (ROOT/'reports/waterscenes_train_inventory.jsonl').open('w',encoding='utf8') as f:
  for r in rows:f.write(json.dumps(r)+'\n')
 result={'source_split':'train','images':len(ids),'native_images_per_class':dict(fc),'native_boxes_per_class':dict(bc),'label_only_images_without_ambiguous_vessels':dict(candidate),'coordinate_hypothesis_object_counts':dict(hyp),'issues':issues,'image_payloads_checked':0,'admitted_images':0,'test_payloads_opened':0,'warning':'label-only counts; not reviewed, deduplicated or temporally thinned; generic ship/boat/vessel quarantined'}
 write_json(ROOT/'reports/waterscenes_train_audit.json',result);print(json.dumps(result))
def audit_existing(old):
 results=[]
 for source,p in [('seaclips',old/'data/SeaClips/seaclips-train.json'),('seadronessee',old/'data/SeaDronesSee_v4/annotations/instances_train.json')]:
  d=json.loads(p.read_text());cats={c['id']:c['name'] for c in d['categories']};ims={i['id']:i for i in d['images']};by=defaultdict(list)
  for a in d['annotations']:by[a['image_id']].append(a)
  fc=Counter();bc=Counter();groupclasses=defaultdict(set);resolved=Counter();resolvedgroups=defaultdict(set)
  mapping={'BOAT':'powered_vessel','SHIP':'powered_vessel','SAILING_VESSEL':'sailboat','LEISURE_VEHICLE':'small_craft','MARINE_MARKER':'navigation_marker','ANIMAL':'unresolved','OBJECT':'unresolved'} if source=='seaclips' else {'ignored':'unresolved','swimmer':'person_in_water','boat':'unresolved','jetski':'powered_vessel','life_saving_appliances':'other_obstacle','buoy':'navigation_marker'}
  for id,im in ims.items():
   ns=[cats[a['category_id']] for a in by[id]];fc.update(set(ns));bc.update(ns)
   group=str(im.get('video_id',im.get('source',{}).get('video','unknown')))
   for n in set(ns):groupclasses[n].add(group)
   mapped=[mapping.get(n,'unresolved') for n in ns]
   if 'unresolved' not in mapped:
    for c in set(mapped)-{'out_of_scope'}:resolved[c]+=1;resolvedgroups[c].add(group)
  results.append({'source':source,'source_annotation':str(p),'annotation_sha256':sha(p.read_bytes()),'source_split':'train','images':len(ims),'native_images_per_class':dict(fc),'native_boxes_per_class':dict(bc),'native_capture_groups_per_class':{k:len(v) for k,v in groupclasses.items()},'label_only_resolvable_images':dict(resolved),'label_only_resolvable_groups':{k:len(v) for k,v in resolvedgroups.items()},'image_payloads_checked':0,'admitted_images':0,'test_payloads_opened':0})
 write_json(ROOT/'reports/existing_train_native_audit.json',results);print(json.dumps(results))
def main():
 p=argparse.ArgumentParser();p.add_argument('source',choices=['debris','waterscenes','existing']);p.add_argument('--old',type=Path,default=Path(r'D:\Projects\maritime-perception'));a=p.parse_args()
 if a.source=='debris':audit_debris()
 elif a.source=='waterscenes':audit_waterscenes()
 else:audit_existing(a.old)
if __name__=='__main__':main()
