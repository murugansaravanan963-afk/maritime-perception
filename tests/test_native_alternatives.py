import unittest,sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from audit_smd import frame_inventory
from acquire_openimages_canoes import eligible_canoe_photo
class NativeAlternativeTests(unittest.TestCase):
 def test_stale_smd_text_cannot_shift_class_assignment(self):
  ids,boxes,issues=frame_inventory({'Object':[3,6],'ObjectType':['Vessel/ship','Ferry','Kayak'],'BB':[[0,0,20,20],[40,30,10,5]]});self.assertEqual(ids,[3,6]);self.assertIn('text_numeric_count_mismatch',issues)
 def test_smd_conflicting_class_name_rejected_even_when_lengths_match(self):
  ids,boxes,issues=frame_inventory({'Object':[6],'ObjectType':'Boat','BB':[0,0,20,20]});self.assertIn('unknown_or_conflicting_native_class',issues)
 def test_single_smd_box_and_class_remain_aligned(self):
  ids,boxes,issues=frame_inventory({'Object':6,'ObjectType':'Kayak','BB':[10,20,30,40]});self.assertEqual(ids,[6]);self.assertEqual(boxes.tolist(),[[10,20,30,40]]);self.assertEqual(issues,[])
 def test_smd_placeholder_and_zero_box_rejected(self):
  ids,boxes,issues=frame_inventory({'Object':0,'ObjectType':[],'BB':[0,0,0,0]});self.assertIn('unknown_native_class_id',issues);self.assertIn('nonfinite_or_degenerate_box',issues)
 def test_depiction_and_group_images_do_not_enter_photo_pool(self):
  normal={'LabelName':'/m/0ph39','IsDepiction':'0','IsGroupOf':'0'}
  self.assertTrue(eligible_canoe_photo({'native_annotations':[normal]}))
  for flag in ['IsDepiction','IsGroupOf']:
   bad={**normal,flag:'1'};self.assertFalse(eligible_canoe_photo({'native_annotations':[normal,bad]}))
 def test_other_object_classes_do_not_create_canoe_candidates(self):
  self.assertFalse(eligible_canoe_photo({'native_annotations':[{'LabelName':'/m/019jd','IsDepiction':'0','IsGroupOf':'0'}]}))
if __name__=='__main__':unittest.main()
