import sys,tempfile,json,unittest,zipfile,io,contextlib
from pathlib import Path
from PIL import Image
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
import audit_candidates as ac
class SourceAuditTests(unittest.TestCase):
 def test_yolo_bom_and_test_payload_separation(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);(r/'raw/dsix').mkdir(parents=True);(r/'reports').mkdir();b=io.BytesIO();Image.new('RGB',(20,10),'blue').save(b,format='JPEG')
   with zipfile.ZipFile(r/'raw/dsix/D_six.zip','w') as z:
    z.writestr('D_six/images/train/a.jpg',b.getvalue());z.writestr('D_six/labels/train/a.txt','\ufeff0 0.5 0.5 0.2 0.4\n');z.writestr('D_six/test/b.jpg',b'corrupt test payload must stay unopened');z.writestr('D_six/images/val/c.jpg',b'invalid validation payload must stay unopened')
   previous=ac.ROOT;ac.ROOT=r
   try:
    with contextlib.redirect_stdout(io.StringIO()):ac.audit_debris()
   finally:ac.ROOT=previous
   report=json.loads((r/'reports/debris_native_audit.json').read_text());row=json.loads((r/'reports/debris_native_manifest.jsonl').read_text());self.assertEqual(report['sources'][0]['images_with_issues'],0);self.assertEqual(row['boxes'][0]['xyxy'],[8.0,3.0,12.0,7.0]);self.assertEqual(report['test_payloads_opened'],0)
 def test_waterbird_context_cannot_be_dropped_as_background(self):
  with tempfile.TemporaryDirectory() as t:
   r=Path(t);(r/'data/SeaClips').mkdir(parents=True);(r/'data/SeaDronesSee_v4/annotations').mkdir(parents=True);(r/'reports').mkdir()
   d={'categories':[{'id':0,'name':'ANIMAL'},{'id':1,'name':'LEISURE_VEHICLE'}],'images':[{'id':1,'video_id':1}],'annotations':[{'image_id':1,'category_id':0},{'image_id':1,'category_id':1}]}
   (r/'data/SeaClips/seaclips-train.json').write_text(json.dumps(d));(r/'data/SeaDronesSee_v4/annotations/instances_train.json').write_text(json.dumps({'categories':[],'images':[],'annotations':[]}));previous=ac.ROOT;ac.ROOT=r
   try:
    with contextlib.redirect_stdout(io.StringIO()):ac.audit_existing(r)
   finally:ac.ROOT=previous
   result=json.loads((r/'reports/existing_train_native_audit.json').read_text());self.assertEqual(result[0]['label_only_resolvable_images'],{})
 def test_invalid_boxes_and_conflicting_duplicates_are_not_clipped(self):
  b=[{'native_class':'floater','xyxy':[0,0,21,10]}]*2;issues=ac.box_issues(b,20,10);self.assertIn('out_of_bounds_or_degenerate:0',issues);self.assertIn('duplicate_box:1',issues);self.assertEqual(b[0]['xyxy'][2],21)
if __name__=='__main__':unittest.main()
