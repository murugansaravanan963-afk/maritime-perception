"""Validated direct reads of original members in a pinned public ZIP index."""
import struct,urllib.request,zlib
from inspect_remote_zip import safe_name

def decode_member(payload,entry):
    safe_name(entry['name'])
    if len(payload)<30:raise ValueError('Truncated ZIP header')
    signature,version,flags,method,mtime,mdate,crc,csize,usize,nlen,elen=struct.unpack('<IHHHHHIIIHH',payload[:30])
    if signature!=0x04034b50 or flags&1:raise ValueError('Invalid or encrypted ZIP member')
    if method not in (0,8):raise ValueError('Unsupported ZIP compression')
    off=30+nlen+elen;end=off+entry['compressed_bytes']
    if len(payload)<end:raise ValueError('Truncated ZIP payload')
    enc='utf8' if flags&0x800 else 'cp437'
    name=payload[30:30+nlen].decode(enc)
    if name!=entry['name']:raise ValueError('ZIP index/header name mismatch')
    if csize==0xffffffff or usize==0xffffffff:
        if version<45:raise ValueError('ZIP64 member requires version45')
        extra=payload[30+nlen:off];pos=0;zip64=None
        while pos<len(extra):
            if pos+4>len(extra):raise ValueError('Truncated ZIP extra header')
            tag,size=struct.unpack_from('<HH',extra,pos);pos+=4
            if pos+size>len(extra):raise ValueError('Truncated ZIP extra field')
            value=extra[pos:pos+size];pos+=size
            if tag==1:
                if zip64 is not None:raise ValueError('Duplicate ZIP64 extra field')
                zip64=value
        if zip64 is None:raise ValueError('Missing ZIP64 size field')
        pos=0
        if usize==0xffffffff:
            if pos+8>len(zip64):raise ValueError('Missing ZIP64 uncompressed size')
            usize=struct.unpack_from('<Q',zip64,pos)[0];pos+=8
        if csize==0xffffffff:
            if pos+8>len(zip64):raise ValueError('Missing ZIP64 compressed size')
            csize=struct.unpack_from('<Q',zip64,pos)[0]
    if not flags&8 and (crc!=int(entry['crc32'],16) or csize!=entry['compressed_bytes'] or usize!=entry['bytes']):
        raise ValueError('ZIP index/header size or CRC mismatch')
    compressed=payload[off:end]
    if method==8:
        obj=zlib.decompressobj(-15);data=obj.decompress(compressed,entry['bytes']+1)
        if not obj.eof or obj.unused_data or obj.unconsumed_tail:raise ValueError('Invalid deflate boundary')
    else:data=compressed
    if len(data)!=entry['bytes'] or format(zlib.crc32(data)&0xffffffff,'08x')!=entry['crc32']:
        raise ValueError('ZIP member length or CRC mismatch')
    return data

def fetch_member(url,archive_bytes,entry):
    start=entry['header_offset'];end=min(archive_bytes-1,start+entry['compressed_bytes']+4095)
    req=urllib.request.Request(url,headers={'User-Agent':'MaritimeDataset/1.0','Range':f'bytes={start}-{end}'})
    with urllib.request.urlopen(req,timeout=45) as resp:
        if resp.status!=206 or resp.headers.get('Content-Range')!=f'bytes {start}-{end}/{archive_bytes}':
            raise ValueError('Server did not honor exact ZIP range')
        payload=resp.read(end-start+2)
    if len(payload)!=end-start+1:raise ValueError('Truncated or oversized HTTP range')
    return decode_member(payload,entry)
