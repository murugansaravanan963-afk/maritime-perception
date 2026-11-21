#!/usr/bin/env python3
"""Read public ZIP metadata/selected annotation files with validated HTTP ranges."""
import argparse,io,json,re,urllib.request,zipfile,hashlib
from pathlib import Path,PurePosixPath
class HTTPRangeFile(io.RawIOBase):
    def __init__(self,url,size):
        self.url=url;self.size=size;self.pos=0;self.cache={};self.bytes_read=0
    def readable(self):return True
    def seekable(self):return True
    def tell(self):return self.pos
    def seek(self,offset,whence=0):
        new=offset if whence==0 else self.pos+offset if whence==1 else self.size+offset
        if new<0:raise ValueError("Negative seek")
        self.pos=new;return new
    def read(self,n=-1):
        if n<0:n=self.size-self.pos
        n=min(n,self.size-self.pos)
        if n<=0:return b""
        result=bytearray()
        while n:
            block=self.pos//1048576;start=block*1048576;end=min(start+1048576,self.size)-1
            if block not in self.cache:
                req=urllib.request.Request(self.url,headers={"User-Agent":"MaritimeDataset/1.0","Range":f"bytes={start}-{end}"})
                with urllib.request.urlopen(req,timeout=35) as resp:
                    expected=f"bytes {start}-{end}/{self.size}"
                    if resp.status!=206 or resp.headers.get("Content-Range")!=expected:
                        raise ValueError("Server did not honor exact range: "+str(resp.headers.get("Content-Range")))
                    data=resp.read()
                if len(data)!=end-start+1:raise ValueError("Truncated range")
                if len(self.cache)>=8:self.cache.pop(next(iter(self.cache)))
                self.cache[block]=data;self.bytes_read+=len(data)
            data=self.cache[block];offset=self.pos-start;take=min(n,len(data)-offset)
            result.extend(data[offset:offset+take]);self.pos+=take;n-=take
        return bytes(result)
def safe_name(name):
    p=PurePosixPath(name)
    if p.is_absolute() or ".." in p.parts or any(":" in x or "\\" in x for x in p.parts):
        raise ValueError("Unsafe ZIP name")
    return p.parts
def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--url",required=True);ap.add_argument("--size",type=int,required=True)
    ap.add_argument("--out",type=Path,required=True);a=ap.parse_args()
    a.out.mkdir(parents=True,exist_ok=True)
    stream=HTTPRangeFile(a.url,a.size);selected=[]
    with zipfile.ZipFile(stream) as z:
        listing=[{"name":i.filename,"bytes":i.file_size,"compressed_bytes":i.compress_size,"crc32":format(i.CRC,"08x")} for i in z.infolist()]
        for i in z.infolist():
            if i.is_dir():continue
            # This tool deliberately does not read any image/video payload.
            if Path(i.filename).suffix.lower() not in (".mat",".pdf",".txt",".xml",".yaml",".csv"):continue
            if i.file_size>50_000_000:continue
            path=a.out.joinpath(*safe_name(i.filename));path.parent.mkdir(parents=True,exist_ok=True)
            data=z.read(i);path.write_bytes(data)
            selected.append({"name":i.filename,"sha256":hashlib.sha256(data).hexdigest(),"bytes":len(data)})
    (a.out/"remote_archive.json").write_text(json.dumps({"url":a.url,"archive_bytes":a.size,
        "downloaded_range_bytes":stream.bytes_read,"members":listing,"selected":selected},indent=2)+"\n")
    print(json.dumps({"archive_bytes":a.size,"downloaded_range_bytes":stream.bytes_read,"selected":selected},indent=2))
if __name__=="__main__":main()

