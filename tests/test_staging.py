import sys,unittest
from pathlib import Path
import numpy as np
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from stage_candidates import panoptic_ids,mask_geometry,native_box
class NativeStagingTests(unittest.TestCase):
 def test_rgb_instance_ids_are_not_semantic_ids(self):
  rgb=np.array([[[15,0,1],[15,0,2]]],dtype=np.uint8)
  self.assertEqual(panoptic_ids(rgb).tolist(),[[65551,131087]])
 def test_two_same_class_instances_remain_separate(self):
  ids=np.array([[65551,0,131087],[65551,0,131087]],dtype=np.uint32)
  ss=[{'id':65551,'area':2,'bbox':[0,0,1,2]},{'id':131087,'area':2,'bbox':[2,0,1,2]}]
  boxes,issues=mask_geometry(ids,ss);self.assertEqual(issues,[]);self.assertEqual(boxes[131087],[2,0,3,2]);self.assertEqual(len(boxes),2)
 def test_incomplete_declared_box_does_not_silently_clip(self):
  ids=np.ones((3,3),dtype=np.uint32);ss=[{'id':1,'area':9,'bbox':[0,0,2,3]}]
  self.assertIn('instance_outside_declared_box:1',mask_geometry(ids,ss)[1])
 def test_coco_xywh_not_inclusive_max(self):
  self.assertEqual(native_box([7,8,1,1]),[7,8,8,9])
 def test_mask_bbox_padding_requires_review(self):
  ids=np.array([[0,0,0],[0,1,0]],dtype=np.uint32)
  ss=[{'id':1,'area':1,'bbox':[0,0,3,2]}]
  self.assertIn('mask_box_not_tight:1',mask_geometry(ids,ss)[1])
if __name__=='__main__':unittest.main()
