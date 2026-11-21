import sys,json,copy,tempfile,unittest,zipfile,io
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import release_gate as rg
from inspect_remote_zip import safe_name
class ReleaseGateTests(unittest.TestCase):
 def setUp(self):
  self.tmp=tempfile.TemporaryDirectory();self.root=Path(self.tmp.name);self.tax=json.dumps({'names':rg.NAMES}).encode();self.th=rg.digest(self.tax)
  self.policy={'primary_viewpoints':['boat','shore'],'allowed_source_splits':{'train':['train','unspecified'],'val':['train','val','unspecified'],'test':['train','val','test','unspecified']},'min_primary_train_images_per_class':1,'min_primary_val_images_per_class':1,'min_primary_test_images_per_class':1,'min_train_capture_groups_per_class':1,'min_eval_capture_groups_per_class':1,'max_train_class_image_ratio':3.0,'max_fraction_of_class_from_one_train_group':1.0}
  self.records=[]
  for i,s in enumerate(['train','val','test']):
   p=self.root/(s+'.png');im=Image.new('RGB',(24,16),(10+i,20,30));im.save(p)
   self.records.append({'id':s,'source':'fixture','source_revision':'test-v1','source_annotation_sha256':'a'*64,'source_split':s,'split':s,'viewpoint':'boat','capture_group':s,'independent_unit':s,'duplicate_cluster':s,'image':{'path':str(p),'width':24,'height':16,'sha256':rg.digest(p.read_bytes()),'pixel_sha256':rg.digest(str(im.size).encode()+im.tobytes())},'boxes':[{'class_id':c,'xyxy':[c,0,c+2,16]} for c in range(6)]})
  self.reviews=self.sign(self.records)
 def tearDown(self):self.tmp.cleanup()
 def sign(self,rs):
  return [{'id':r['id'],'record_sha256':rg.canonical_hash(r),'taxonomy_sha256':self.th,'reviewed_by':'test fixture only','reviewed_at':'2026-10-22','approved':True,'full_frame_complete':True,'class_and_box_reviewed':True,'capture_group_verified':True,'near_duplicate_audit_complete':True,'source_terms_verified':True} for r in rs]
 def check(self,records=None,reviews=None):return rg.gate(records or self.records,reviews if reviews is not None else self.reviews,self.policy,self.th)
 def test_valid_and_roundtrip_export(self):
  self.assertTrue(self.check()['passed']);dest=self.root/'release';rg.publish(self.records,self.reviews,self.policy,self.tax,dest)
  self.assertEqual(len(list(dest.rglob('*.png'))),3)
  for line in next((dest/'labels/train').glob('*.txt')).read_text().splitlines():
   c,x,y,w,h=map(float,line.split());self.assertAlmostEqual((x-w/2)*24,c,places=7);self.assertAlmostEqual(h*16,16,places=7)
 def test_changed_boxes_invalidate_review(self):
  r=copy.deepcopy(self.records);r[0]['boxes'][0]['xyxy'][2]=4;self.assertIn('train:review_record_sha256_mismatch',self.check(r)['errors'])
 def test_corrupt_image_blocked(self):
  Path(self.records[0]['image']['path']).write_bytes(b'not a PNG');self.assertFalse(self.check()['passed'])
 def test_semantically_unresolved_blocked(self):
  r=copy.deepcopy(self.records);r[0]['issues']=['unresolved_boat'];self.assertIn('train:unresolved_audit_issues',self.check(r,self.sign(r))['errors'])
 def test_unresolved_native_objects_cannot_hide_behind_empty_issues(self):
  r=copy.deepcopy(self.records);r[0]['unresolved_objects']=[{'native_class':'boat','native_xyxy':[1,1,5,5]}];r[0]['issues']=[]
  self.assertIn('train:unresolved_native_objects',self.check(r,self.sign(r))['errors'])
 def test_pending_checks_cannot_be_overridden_by_bulk_review_flags(self):
  r=copy.deepcopy(self.records);r[0]['checks_pending']=['person_in_water_context']
  self.assertIn('train:pending_dataset_checks',self.check(r,self.sign(r))['errors'])
 def test_out_of_range_class_and_box(self):
  r=copy.deepcopy(self.records);r[0]['boxes'][0]={'class_id':6,'xyxy':[0,0,2,3]};r[0]['boxes'][1]['xyxy']=[-1,0,2,17];e=self.check(r,self.sign(r))['errors'];self.assertIn('train:invalid_class',e);self.assertIn('train:box_outside_image_or_degenerate',e)
 def test_duplicate_pixels_not_counted(self):
  r=copy.deepcopy(self.records);r[1]['image']=copy.deepcopy(r[0]['image']);self.assertIn('val:duplicate_image_pixel_sha256',self.check(r,self.sign(r))['errors'])
 def test_capture_group_cannot_cross_splits(self):
  r=copy.deepcopy(self.records);r[1]['capture_group']='train';self.assertIn('val:capture_group_crosses_splits',self.check(r,self.sign(r))['errors'])
 def test_near_duplicate_cluster_cannot_cross_splits(self):
  r=copy.deepcopy(self.records);r[1]['duplicate_cluster']='train';self.assertIn('val:duplicate_cluster_crosses_splits',self.check(r,self.sign(r))['errors'])
 def test_official_test_cannot_train(self):
  r=copy.deepcopy(self.records);r[0]['source_split']='test';self.assertIn('train:publisher_split_violation',self.check(r,self.sign(r))['errors'])
 def test_missing_fullframe_review_blocked(self):
  v=copy.deepcopy(self.reviews);v[0]['full_frame_complete']=False;self.assertFalse(self.check(reviews=v)['passed'])
 def test_aerial_cannot_fill_surface_quota(self):
  r=copy.deepcopy(self.records);r[0]['viewpoint']='aerial';self.assertIn('coverage:train:person_in_water:0<1',self.check(r,self.sign(r))['errors'])
 def test_total_training_quota_is_separate_from_surface_quota(self):
  self.policy['min_train_images_per_class']=2
  result=self.check()
  self.assertIn('total_training_coverage:person_in_water:1<2',result['errors'])
  self.assertFalse(any(e.startswith('coverage:train:') for e in result['errors']))
 def test_aerial_counts_in_total_but_never_surface_coverage(self):
  r=copy.deepcopy(self.records);r[0]['viewpoint']='aerial';self.policy['min_train_images_per_class']=1
  result=self.check(r,self.sign(r))
  self.assertEqual(result['total_images_per_class']['train']['person_in_water'],1)
  self.assertEqual(result['primary_images_per_class']['train']['person_in_water'],0)
  self.assertIn('coverage:train:person_in_water:0<1',result['errors'])
 def test_primary_fraction_cannot_be_filled_by_aerial(self):
  r=copy.deepcopy(self.records);r[0]['viewpoint']='aerial';self.policy['min_primary_train_images_per_class']=0;self.policy['min_train_capture_groups_per_class']=0
  self.policy['min_primary_fraction_of_training_images']=0.6
  result=self.check(r,self.sign(r))
  self.assertIn('insufficient_primary_training_image_fraction',result['errors'])
 def test_balanced_totals_do_not_waive_fullframe_checks(self):
  self.policy['min_train_images_per_class']=1;self.policy['min_primary_fraction_of_training_images']=0.6
  v=copy.deepcopy(self.reviews);v[0]['full_frame_complete']=False
  self.assertFalse(self.check(reviews=v)['passed'])
 def test_surface_subtype_requires_explicit_basis(self):
  r=copy.deepcopy(self.records);r[0]['viewpoint']='surface';self.policy['primary_viewpoints'].append('surface')
  self.assertIn('train:missing_surface_viewpoint_evidence',self.check(r,self.sign(r))['errors'])
  r[0]['viewpoint_basis']={'surface_camera_confirmed':True,'reason':'Fixture only: confirmed surface camera; exact support unresolved.'}
  self.assertTrue(self.check(r,self.sign(r))['passed'])
 def test_empty_manifest_cannot_publish(self):
  dest=self.root/'badrelease'
  with self.assertRaises(ValueError):rg.publish([],[],self.policy,self.tax,dest)
  self.assertFalse(dest.exists())
 def test_duplicate_boxes_rejected(self):
  r=copy.deepcopy(self.records);r[0]['boxes'].append(copy.deepcopy(r[0]['boxes'][0]));self.assertIn('train:duplicate_box',self.check(r,self.sign(r))['errors'])
 def test_augmented_examples_not_accepted_as_coverage(self):
  r=copy.deepcopy(self.records);r[0]['augmented_from']='parent';self.assertFalse(self.check(r,self.sign(r))['passed'])
 def test_unsafe_zip_member_paths_rejected(self):
  for n in ['../escape.jpg','C:/escape.jpg','/escape.jpg','x\\..\\escape.jpg']:
   with self.assertRaises(ValueError):safe_name(n)
if __name__=='__main__':unittest.main(verbosity=2)
