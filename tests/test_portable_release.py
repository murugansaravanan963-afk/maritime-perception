import sys,json,copy,shutil,tempfile,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
sys.path.insert(0,str(Path(__file__).resolve().parent))
import test_release_gate as release_fixture
from preflight_release import verify_release,runtime_yaml
import release_gate as rg
from merge_candidates import exact_clusters,corrected_candidates
class PortableReleaseTests(unittest.TestCase):
 def setUp(self):
  self.fixture=release_fixture.ReleaseGateTests();self.fixture.setUp();self.root=self.fixture.root;self.dest=self.root/'export';rg.publish(self.fixture.records,self.fixture.reviews,self.fixture.policy,self.fixture.tax,self.dest)
 def tearDown(self):self.fixture.tearDown()
 def test_moved_release_uses_exported_images_not_original_paths(self):
  moved=self.root/'sol_copy';shutil.copytree(self.dest,moved)
  for r in self.fixture.records:Path(r['image']['path']).unlink()
  self.assertEqual(verify_release(moved)['images_verified'],3)
  yp=runtime_yaml(moved,self.root/'runtime/data.yaml');self.assertIn(str(moved).replace('\\','\\\\'),yp.read_text())
 def test_changed_exported_label_blocked(self):
  p=next((self.dest/'labels/train').glob('*.txt'));p.write_text('0 .5 .5 .1 .1\n')
  with self.assertRaisesRegex(ValueError,'count changed'):verify_release(self.dest)
 def test_changed_exported_image_blocked(self):
  next((self.dest/'images/train').glob('*.png')).write_bytes(b'broken')
  with self.assertRaises(Exception):verify_release(self.dest)
 def test_loosened_policy_invalidates_fingerprint(self):
  (self.dest/'release_policy.json').write_text('{}')
  with self.assertRaisesRegex(ValueError,'Changed release metadata'):verify_release(self.dest)
 def test_candidates_without_release_fingerprint_are_blocked(self):
  with self.assertRaisesRegex(ValueError,'candidate inventories cannot be trained'):verify_release(self.root/'unreleased')
 def test_annotation_correction_cannot_replace_reviewed_image(self):
  base=copy.deepcopy(self.fixture.records[0]);corrected=copy.deepcopy(base);corrected['image']['sha256']='f'*64
  patch={'id':base['id'],'parent_record_sha256':rg.canonical_hash(base),'corrected_record':corrected}
  with self.assertRaisesRegex(ValueError,'protected provenance'):corrected_candidates([base],[patch])
 def test_annotation_correction_requires_unchanged_parent(self):
  base=copy.deepcopy(self.fixture.records[0]);patch={'id':base['id'],'parent_record_sha256':'0'*64,'corrected_record':base}
  with self.assertRaisesRegex(ValueError,'parent changed'):corrected_candidates([base],[patch])
 def test_ambiguous_duplicate_family_quarantines_both_members(self):
  rs=copy.deepcopy(self.fixture.records[:2]);rs[1]['image']=rs[0]['image'];rs[1]['boxes'][0]['xyxy']=[0,0,3,16]
  for r in rs:r['issues']=[]
  families=exact_clusters(rs);self.assertTrue(families[0]['annotation_conflict']);self.assertTrue(all('exact_duplicate_annotation_conflict' in r['issues'] for r in rs))
if __name__=='__main__':unittest.main()
