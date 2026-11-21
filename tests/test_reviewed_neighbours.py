import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_reviewed_neighbours import neighbour_pairs,verified_feature
from audit_candidates import image_info
from PIL import Image
import tempfile,io
def f(n,d,p,g='same',pix=None):
 return dict(id=str(n),dhash64=format(d,'016x'),phash64=format(p,'016x'),split_guard_group=g,pixel_sha256=pix or str(n))
class NeighboursTests(unittest.TestCase):
 def test_no_per_image_cap(self):
  pairs=list(neighbour_pairs([f(n,n,n) for n in range(8)]))
  self.assertEqual(len(pairs),28)
  self.assertTrue(all(not p['cross_guard'] for p in pairs))
 def test_either_hash_is_sufficient(self):
  self.assertEqual(len(list(neighbour_pairs([f(1,0,0),f(2,0,(1<<63)-1)]))),1)
  self.assertEqual(len(list(neighbour_pairs([f(1,0,0),f(2,(1<<63)-1,0)]))),1)
 def test_distant_pair_is_not_a_hint(self):
  self.assertEqual(list(neighbour_pairs([f(1,0,0),f(2,(1<<64)-1,(1<<64)-1)])),[])
 def test_cross_guard_and_exact_pixel_flags(self):
  pair=list(neighbour_pairs([f(1,1,1,'A','x'),f(2,1,1,'B','x')]))[0]
  self.assertTrue(pair['cross_guard']);self.assertTrue(pair['exact_pixels'])
class FeatureIntegrityTests(unittest.TestCase):
 def fixture(self,folder):
  p=Path(folder)/'photo.png';Image.new('RGB',(23,17),'blue').save(p)
  info=image_info(p.read_bytes())
  return dict(id='test',split_guard_group='capture',image=dict(path=str(p),sha256=info['image_sha256'],pixel_sha256=info['pixel_sha256'],dhash64=info['dhash64']))
 def test_image_and_size_prefixed_pixels_verified(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.fixture(d);result=verified_feature(r)
   self.assertEqual(result['image_sha256'],r['image']['sha256'])
 def test_changed_decoded_pixel_hash_rejected(self):
  with tempfile.TemporaryDirectory() as d:
   r=self.fixture(d);r['image']['pixel_sha256']='0'*64
   with self.assertRaises(ValueError):verified_feature(r)
if __name__=='__main__':unittest.main()
