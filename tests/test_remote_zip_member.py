import io,zipfile,zlib,unittest,sys,copy,struct
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'scripts'))
from remote_zip_member import decode_member
class RemoteZipMemberIntegrity(unittest.TestCase):
 def fixture(self,compression=zipfile.ZIP_DEFLATED):
  bio=io.BytesIO()
  with zipfile.ZipFile(bio,'w',compression=compression) as z:z.writestr('images/frame.PNG',b'original image payload'*300)
  b=bio.getvalue()
  with zipfile.ZipFile(io.BytesIO(b)) as z:
   i=z.getinfo('images/frame.PNG');e={'name':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,'crc32':format(i.CRC,'08x'),'header_offset':i.header_offset}
  return b,e
 def test_original_deflated_bytes(self):
  b,e=self.fixture();self.assertEqual(decode_member(b,e),b'original image payload'*300)
 def test_original_stored_bytes(self):
  b,e=self.fixture(zipfile.ZIP_STORED);self.assertEqual(decode_member(b,e),b'original image payload'*300)
 def test_truncated_payload_rejected(self):
  b,e=self.fixture()
  with self.assertRaises(ValueError):decode_member(b[:40],e)
 def test_mismatched_index_name_rejected(self):
  b,e=self.fixture();e['name']='images/different.PNG'
  with self.assertRaises(ValueError):decode_member(b,e)
 def test_corrupt_payload_rejected(self):
  b,e=self.fixture(zipfile.ZIP_STORED);b=bytearray(b);b[70]^=0xff
  with self.assertRaises(ValueError):decode_member(bytes(b),e)
 def test_index_crc_change_rejected(self):
  b,e=self.fixture();e['crc32']='00000000'
  with self.assertRaises(ValueError):decode_member(b,e)
 def test_unsafe_archive_member_rejected(self):
  b,e=self.fixture();e['name']='../outside.PNG'
  with self.assertRaises(ValueError):decode_member(b,e)
 def test_declared_output_limit_rejected(self):
  b,e=self.fixture();e['bytes']=10
  with self.assertRaises(ValueError):decode_member(b,e)

 def zip64_fixture(self):
  bio=io.BytesIO()
  with zipfile.ZipFile(bio,'w',compression=zipfile.ZIP_DEFLATED) as z:
   with z.open('images/frame.PNG','w',force_zip64=True) as f:f.write(b'original image payload'*300)
  b=bio.getvalue()
  with zipfile.ZipFile(io.BytesIO(b)) as z:
   i=z.getinfo('images/frame.PNG');e={'name':i.filename,'bytes':i.file_size,'compressed_bytes':i.compress_size,'crc32':format(i.CRC,'08x'),'header_offset':i.header_offset}
  return b,e
 def test_original_zip64_local_header_with_small_image(self):
  b,e=self.zip64_fixture();self.assertEqual(decode_member(b,e),b'original image payload'*300)
 def test_zip64_size_disagreement_rejected(self):
  b,e=self.zip64_fixture();b=bytearray(b);nlen=struct.unpack_from('<H',b,26)[0];b[30+nlen+4]^=1
  with self.assertRaises(ValueError):decode_member(bytes(b),e)
 def test_missing_zip64_sizes_rejected(self):
  b,e=self.zip64_fixture();b=bytearray(b);nlen,elen=struct.unpack_from('<HH',b,26);del b[30+nlen:30+nlen+elen];struct.pack_into('<H',b,28,0)
  with self.assertRaises(ValueError):decode_member(bytes(b),e)

if __name__=='__main__':unittest.main()
