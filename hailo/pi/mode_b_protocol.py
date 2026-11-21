#!/usr/bin/env python3
"""Y8PP Mode B protocol (Pi <-> FPGA), shared by pi_mode_b_client.py and mock_virtual_fpga.py.

Request : 16 B header + 68 B per candidate + CRC16          (same bytes for TCP now and SPI later)
Reply   : status, frame_id, n_det, 2 cycle counters + 12 B per detection + CRC16
TCP only: every message is preceded by a 4 B big-endian length (not part of the SPI protocol).
SPI (DE10-Lite): packet in one CS-low transfer, then a separate read transfer whose first byte is a dummy.
All multi-byte fields are big-endian. CRC = CRC-16/CCITT-FALSE (poly 0x1021, init 0xFFFF).
Matches the RTL team's mode_b_v1 parser/result builder and virtual_fpga_mode_b client.
"""
import socket,struct,time
import numpy as np

MAGIC=b'Y8PP';VERSION=1;CMD_FRAME_CANDIDATES=0x03
NUM_CLASSES=6;BINS=16;RECORD_BYTES=4+4*BINS;DET_BYTES=12;HEADER_BYTES=16;REPLY_HEADER_BYTES=12
SCORE_Q_THR=64;IOU_NUM,IOU_DEN=9,20;MAX_CAND=4096;MAX_DET=100      # fixed in the RTL (rtl_params.vh)
STATUS_BITS={0:'ok',1:'crc_error',2:'format_error',3:'overflow',4:'max_det'}

def _crc_table():
    t=[]
    for b in range(256):
        c=b<<8
        for _ in range(8):c=((c<<1)^0x1021)&0xFFFF if c&0x8000 else (c<<1)&0xFFFF
        t.append(c)
    return t
_CRC=_crc_table()

def crc16(data):
    c=0xFFFF
    for b in data:c=((c<<8)&0xFFFF)^_CRC[(c>>8)^b]
    return c

def prefilter(bbox_q,score_q,q_thr=SCORE_Q_THR):
    """Pi side of mode B: per anchor argmax class (lowest index wins ties) and max score; keep score >= q_thr."""
    cls=score_q.argmax(1);best=score_q.max(1);keep=np.nonzero(best>=q_thr)[0]
    return keep,cls[keep],best[keep]

def build_packet(bbox_q,score_q,frame_id,q_thr=SCORE_Q_THR):
    """(N,64) + (N,6) uint8 Hailo outputs -> (packet bytes, number of candidates)."""
    keep,cls,best=prefilter(bbox_q,score_q,q_thr)
    rec=np.empty((len(keep),RECORD_BYTES),np.uint8)
    rec[:,0]=keep>>8;rec[:,1]=keep&0xFF;rec[:,2]=cls;rec[:,3]=best;rec[:,4:]=bbox_q[keep]
    payload=rec.tobytes()
    head=MAGIC+struct.pack('>BBHHBBI',VERSION,CMD_FRAME_CANDIDATES,frame_id&0xFFFF,len(keep),NUM_CLASSES,BINS,len(payload))
    body=head+payload
    return body+struct.pack('>H',crc16(body)),len(keep)

def parse_packet(packet):
    """Inverse of build_packet (used by the mock FPGA). Returns frame_id, anchor, cls, score, bbox (n,64), error."""
    if len(packet)<HEADER_BYTES+2:return None,'format_error: short packet'
    magic,ver,cmd,fid,n,nc,bins,plen=struct.unpack('>4sBBHHBBI',packet[:HEADER_BYTES])
    if magic!=MAGIC or ver!=VERSION or cmd!=CMD_FRAME_CANDIDATES or nc!=NUM_CLASSES or bins!=BINS or plen!=n*RECORD_BYTES \
       or len(packet)!=HEADER_BYTES+plen+2:
        return None,'format_error'
    if struct.unpack('>H',packet[-2:])[0]!=crc16(packet[:-2]):return None,'crc_error'
    rec=np.frombuffer(packet[HEADER_BYTES:-2],np.uint8).reshape(n,RECORD_BYTES)
    anchor=(rec[:,0].astype(np.int64)<<8)|rec[:,1]
    return dict(frame_id=fid,anchor=anchor,cls=rec[:,2].astype(np.int64),score=rec[:,3].astype(np.int64),bbox=rec[:,4:]),None

def build_reply(frame_id,dets,status=0x01,cycles_first=0,cycles_last=0):
    """dets: list of (x1,y1,x2,y2,cls,score,anchor) with coordinates in 1/16 px."""
    body=struct.pack('>BHBII',status,frame_id&0xFFFF,len(dets),cycles_first,cycles_last)
    for x1,y1,x2,y2,c,s,a in dets:body+=struct.pack('>HHHHBBH',x1,y1,x2,y2,c,s,a)
    return body+struct.pack('>H',crc16(body))

def parse_reply(reply):
    """Reply bytes -> dict. Raises ValueError on a malformed reply or bad CRC."""
    if len(reply)<REPLY_HEADER_BYTES+2:raise ValueError(f'reply too short: {len(reply)} B')
    status,fid,n,c1,c2=struct.unpack('>BHBII',reply[:REPLY_HEADER_BYTES])
    if len(reply)!=REPLY_HEADER_BYTES+DET_BYTES*n+2:raise ValueError(f'reply length {len(reply)} != {REPLY_HEADER_BYTES+DET_BYTES*n+2} for n_det={n}')
    rx,calc=struct.unpack('>H',reply[-2:])[0],crc16(reply[:-2])
    if rx!=calc:raise ValueError(f'reply CRC 0x{rx:04X} != calculated 0x{calc:04X}')
    dets=[struct.unpack('>HHHHBBH',reply[REPLY_HEADER_BYTES+DET_BYTES*i:REPLY_HEADER_BYTES+DET_BYTES*(i+1)]) for i in range(n)]
    flags=[name for bit,name in STATUS_BITS.items() if status>>bit&1]
    return dict(status=status,flags=flags,frame_id=fid,n_det=n,cycles_first=c1,cycles_last=c2,dets=[tuple(int(v) for v in d) for d in dets])

def recv_exact(sock,n):
    data=bytearray()
    while len(data)<n:
        chunk=sock.recv(n-len(data))
        if not chunk:raise ConnectionError('connection closed early')
        data.extend(chunk)
    return bytes(data)

def send_msg(sock,data):sock.sendall(struct.pack('>I',len(data))+data)

def recv_msg(sock,limit=10_000_000):
    n=struct.unpack('>I',recv_exact(sock,4))[0]
    if n>limit:raise ValueError(f'message too large: {n} B')
    return recv_exact(sock,n) if n else b''

def exchange(host,port,packet,timeout):
    """One request per connection, like the RTL team's bridge. Returns reply bytes (b'' = bridge failure)."""
    with socket.create_connection((host,port),timeout=timeout) as s:
        s.settimeout(timeout);send_msg(s,packet);return recv_msg(s)

SPI_MAX_XFER=4096          # py-spidev xfer2 limit; one xfer2 keeps CS low for the whole packet
SPI_READ_BYTES=1+REPLY_HEADER_BYTES+DET_BYTES*MAX_DET+2   # dummy byte + longest possible reply

def spi_exchange(spi,packet,wait_s):
    """DE10-Lite SPI flow: write the packet in one CS-low transfer, wait, then read 1 dummy byte + reply.
    The reply length depends on n_det, so read the longest possible reply and cut it using the n_det byte.
    Returns (reply bytes, {'spi_write_ms','spi_read_ms'})."""
    if len(packet)>SPI_MAX_XFER:raise ValueError(f'packet {len(packet)} B > {SPI_MAX_XFER} B spidev limit')
    t0=time.perf_counter();spi.xfer2(list(packet));t1=time.perf_counter()
    time.sleep(wait_s)
    t2=time.perf_counter();raw=bytes(spi.xfer2([0]*SPI_READ_BYTES))[1:];t3=time.perf_counter()
    n=raw[3]
    if n>MAX_DET:raise ValueError(f'n_det {n} > {MAX_DET}: no valid reply (FPGA not ready or wiring)')
    return raw[:REPLY_HEADER_BYTES+DET_BYTES*n+2],{'spi_write_ms':(t1-t0)*1e3,'spi_read_ms':(t3-t2)*1e3}

def golden_dets(gm,model,params,bbox_q,score_q):
    """Bit-exact expected FPGA output from golden_model.fixed_ref, in reply order (class asc, score desc, anchor asc)."""
    r=gm.fixed_ref(model,bbox_q,score_q,params)
    dets=[(int(r['x1'][i]),int(r['y1'][i]),int(r['x2'][i]),int(r['y2'][i]),int(r['cls'][i]),int(r['sq'][i]),int(i)) for i in r['dets']]
    return dets,r['n_cand'],r['overflow']
