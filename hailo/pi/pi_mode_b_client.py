#!/usr/bin/env python3
"""Send real Hailo frames to the (virtual) FPGA in Mode B and check every reply bit-exactly.

Pi + Hailo -> prefilter (argmax, score >= 64) -> Y8PP packet -> TCP -> FPGA bridge (ModelSim RTL)
           -> reply -> CRC/status check -> compare with golden_model.fixed_ref on the same uint8 outputs.

  # live: run the HEF on test images (on the Pi)
  python3 pi_mode_b_client.py --hef maritime6_960.hef --images test/images --host <PC address> --frames 30
  # offline: replay frames saved by dump_vectors.py (no Hailo needed; also works on the laptop)
  python3 pi_mode_b_client.py --vectors fpga_vectors --host 127.0.0.1
  # real FPGA board (DE10-Lite) on the Pi's SPI0 CE0 pins instead of TCP
  python3 pi_mode_b_client.py --hef maritime6_960.hef --images test/images --spi 0.0 --spi-hz 1000000

Times written to results.json:
  pi_prefilter_pack_ms  numpy prefilter + packing on this machine (the C++ version is ~0.11 ms)
  round_trip_ms         TCP: TCP + ModelSim wall time, NOT FPGA latency. SPI: real write + wait + read time.
  spi_write_ms/read_ms  SPI only: measured time of the two SPI transfers
  fpga_us               RTL cycles from last request byte to done / --fclk-mhz (the real FPGA compute time)
  spi_ms_10/20MHz       request + reply bytes over SPI at that clock (estimate)
"""
import argparse,json,sys,time
from pathlib import Path
import numpy as np
import cv2
import maritime_post as mp
import golden_model as gm
import mode_b_protocol as pb

def hailo_frames(hef_path,images,limit):
    """Yields (name, letterboxed RGB, bbox_q (N,64) uint8, score_q (N,6) uint8, quant) like dump_vectors.py."""
    from hailo_platform import HEF,VDevice,FormatType,HailoStreamInterface,ConfigureParams,InputVStreamParams,OutputVStreamParams,InferVStreams
    hef=HEF(hef_path);ins=hef.get_input_vstream_infos();outs=hef.get_output_vstream_infos()
    sizes={o.name:int(np.prod(o.shape)) for o in outs}
    bo=next(o for o in outs if sizes[o.name]==mp.N_ANCHORS*64);so=next(o for o in outs if sizes[o.name]==mp.N_ANCHORS*6)
    quant=dict(bbox_scale=float(bo.quant_info.qp_scale),bbox_zp=float(bo.quant_info.qp_zp),
               score_scale=float(so.quant_info.qp_scale),score_zp=float(so.quant_info.qp_zp),qbits=8,sbits=8)
    files=sorted(p for p in Path(images).iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png','.bmp'))
    files=files[:limit] if limit else files
    with VDevice(VDevice.create_params()) as dev:
        ng=dev.configure(hef,ConfigureParams.create_from_hef(hef,interface=HailoStreamInterface.PCIe))[0]
        ip=InputVStreamParams.make(ng,format_type=FormatType.UINT8);op=OutputVStreamParams.make(ng,format_type=FormatType.AUTO)
        with ng.activate(),InferVStreams(ng,ip,op) as pipe:
            for p in files:
                rgb,_,_=mp.letterbox(cv2.imread(str(p)))
                r=pipe.infer({ins[0].name:np.ascontiguousarray(rgb[None])})
                bq=np.asarray(r[bo.name]).reshape(mp.N_ANCHORS,64);sq=np.asarray(r[so.name]).reshape(mp.N_ANCHORS,6)
                if bq.dtype!=np.uint8 or sq.dtype!=np.uint8:raise SystemExit(f'Hailo outputs are {bq.dtype}/{sq.dtype}, expected uint8')
                yield p.name,rgb,bq,sq,quant

def vector_frames(vec_dir,limit):
    """Yields frames saved by dump_vectors.py (frame_XXXX.npz + .png + meta.json)."""
    quant,_,frames=gm.load_real(str(vec_dir))
    for name,bq,sq in (frames[:limit] if limit else frames):
        png=Path(vec_dir)/f'{name}.png';rgb=cv2.cvtColor(cv2.imread(str(png)),cv2.COLOR_BGR2RGB) if png.exists() else None
        yield name,rgb,bq.astype(np.uint8),sq.astype(np.uint8),quant

def draw(rgb,dets,path):
    img=cv2.cvtColor(rgb,cv2.COLOR_RGB2BGR)
    for x1,y1,x2,y2,c,s,_ in dets:
        p1,p2=(x1//16,y1//16),(x2//16,y2//16)
        cv2.rectangle(img,p1,p2,(0,255,0),2)
        cv2.putText(img,f'{mp.NAMES[c]} {s/255:.2f}',(p1[0],max(p1[1]-6,12)),cv2.FONT_HERSHEY_SIMPLEX,0.6,(0,255,0),2)
    cv2.imwrite(str(path),img)

def stats(x):
    x=np.asarray(x,float)
    return {'n':int(x.size),'median':float(np.median(x)),'p95':float(np.percentile(x,95)),'max':float(x.max())} if x.size else None

def main():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    src=ap.add_mutually_exclusive_group(required=True)
    src.add_argument('--hef',help='run this HEF on --images (Pi + Hailo)');src.add_argument('--vectors',type=Path,help='replay dump_vectors.py frames')
    ap.add_argument('--images',type=Path,help='image folder for --hef');ap.add_argument('--frames',type=int,default=0,help='0 = all')
    ap.add_argument('--host',default='127.0.0.1');ap.add_argument('--port',type=int,default=9000)
    ap.add_argument('--timeout',type=float,default=600,help='seconds per frame (ModelSim recompiles for every packet)')
    ap.add_argument('--spi',metavar='BUS.DEV',help='use the real FPGA on /dev/spidevBUS.DEV (e.g. 0.0) instead of TCP')
    ap.add_argument('--spi-hz',type=int,default=1_000_000);ap.add_argument('--spi-wait-us',type=int,default=2000,help='wait between packet write and reply read')
    ap.add_argument('--fclk-mhz',type=float,default=None,help='FPGA core clock (default 100 with --spi, 150 for ModelSim)');ap.add_argument('--conf',type=float,default=0.25);ap.add_argument('--iou',type=float,default=0.45)
    ap.add_argument('--send-empty',action='store_true',help='also send frames with 0 candidates (RTL V1 does not handle them yet)')
    ap.add_argument('--overlays',type=int,default=10,help='save this many images with the FPGA detections drawn')
    ap.add_argument('--out',type=Path,default=Path('mode_b_results'))
    a=ap.parse_args()
    if a.hef and not a.images:ap.error('--hef needs --images')
    a.out.mkdir(parents=True,exist_ok=True)
    model=gm.Model(mp.SIZE,mp.SIZE,pb.NUM_CLASSES,pb.MAX_CAND,pb.MAX_DET)
    frames=hailo_frames(a.hef,a.images,a.frames) if a.hef else vector_frames(a.vectors,a.frames)
    fclk=a.fclk_mhz or (100.0 if a.spi else 150.0)
    if a.spi:
        import spidev
        bus,dev=(int(v) for v in a.spi.split('.'))
        spi=spidev.SpiDev();spi.open(bus,dev)
        spi.mode=0;spi.max_speed_hz=a.spi_hz;spi.bits_per_word=8;spi.lsbfirst=False
        link=f'/dev/spidev{a.spi} at {a.spi_hz/1e6:g} MHz'
        send=lambda p:pb.spi_exchange(spi,p,a.spi_wait_us/1e6)
    else:
        link=f'{a.host}:{a.port}'
        send=lambda p:(pb.exchange(a.host,a.port,p,a.timeout),{})

    rows=[];params=None
    for k,(name,rgb,bq,sq,quant) in enumerate(frames):
        if params is None:
            params=gm.make_params(quant,a.conf,a.iou)
            if (params['q_thr'],params['iou_num'],params['iou_den'])!=(pb.SCORE_Q_THR,pb.IOU_NUM,pb.IOU_DEN):
                sys.exit(f"HEF quant gives score threshold {params['q_thr']}, IoU {params['iou_num']}/{params['iou_den']}, "
                         f'but the RTL is built for {pb.SCORE_Q_THR}, {pb.IOU_NUM}/{pb.IOU_DEN}. Regenerate rtl_params.vh first.')
            print(f"[client] score threshold q>={params['q_thr']}  IoU {params['iou_num']}/{params['iou_den']}  "
                  f"LUT depth {len(params['lut'])}  link {link}",flush=True)
        fid=k&0xFFFF
        t=time.perf_counter();packet,n_cand=pb.build_packet(bq,sq,fid,params['q_thr']);t_pack=(time.perf_counter()-t)*1e3
        expect,g_cand,g_overflow=pb.golden_dets(gm,model,params,bq,sq)
        row=dict(frame=name,frame_id=fid,candidates=n_cand,packet_bytes=len(packet),expected_dets=len(expect),pi_prefilter_pack_ms=t_pack)
        if n_cand!=g_cand:row.update(result='ERROR',error=f'prefilter kept {n_cand}, golden {g_cand}')
        elif n_cand==0 and not a.send_empty:row.update(result='SKIPPED',note='0 candidates (not sent)')
        else:
            try:
                t=time.perf_counter();raw,link_times=send(packet);row['round_trip_ms']=(time.perf_counter()-t)*1e3
                row.update(link_times)
                if not raw:raise ValueError('bridge returned an empty reply (ModelSim failed)')
                rep=pb.parse_reply(raw)
                problems=[]
                if rep['frame_id']!=fid:problems.append(f"frame_id {rep['frame_id']} != {fid}")
                if 'ok' not in rep['flags'] or {'crc_error','format_error'}&set(rep['flags']):problems.append(f"status {rep['flags']}")
                if rep['dets']!=expect:
                    problems.append('detections differ from golden'+(' (same set, different order)' if sorted(rep['dets'])==sorted(expect) else ''))
                row.update(result='MISMATCH' if problems else 'PASS',problems=problems,status=rep['flags'],fpga_dets=rep['dets'],
                           reply_bytes=len(raw),cycles_first=rep['cycles_first'],cycles_last=rep['cycles_last'],
                           fpga_us=rep['cycles_last']/fclk,spi_ms_10MHz=(len(packet)+len(raw))*8/10e3,spi_ms_20MHz=(len(packet)+len(raw))*8/20e3)
                if not problems and rgb is not None and k<a.overlays:draw(rgb,rep['dets'],a.out/f'{Path(name).stem}_fpga.jpg')
            except Exception as e:
                row.update(result='ERROR',error=f'{type(e).__name__}: {e}')
        if row['result']!='PASS':row['expected']=expect
        rows.append(row)
        extra=f"dets {row['expected_dets']}  fpga {row['fpga_us']:.1f} us  trip {row['round_trip_ms']:.1f} ms" if row['result']=='PASS' else row.get('error') or row.get('note') or row.get('problems')
        print(f"[client] {name}: cand {n_cand:4d}  {len(packet):6d} B  {row['result']:8s} {extra}",flush=True)

    passed=[r for r in rows if r['result']=='PASS'];bad=[r for r in rows if r['result'] in ('MISMATCH','ERROR')]
    summary=dict(link=link,source=a.hef or str(a.vectors),frames=len(rows),passed=len(passed),
                 mismatched=sum(r['result']=='MISMATCH' for r in rows),errors=sum(r['result']=='ERROR' for r in rows),
                 skipped_empty=sum(r['result']=='SKIPPED' for r in rows),
                 candidates=stats([r['candidates'] for r in rows]),packet_bytes=stats([r['packet_bytes'] for r in rows]),
                 pi_prefilter_pack_ms=stats([r['pi_prefilter_pack_ms'] for r in rows]),fpga_us=stats([r['fpga_us'] for r in passed]),
                 spi_ms_10MHz=stats([r['spi_ms_10MHz'] for r in passed]),spi_ms_20MHz=stats([r['spi_ms_20MHz'] for r in passed]),
                 fclk_mhz=fclk)
    summary['round_trip_ms' if a.spi else 'round_trip_ms_NOT_FPGA_LATENCY']=stats([r['round_trip_ms'] for r in passed])
    if a.spi:summary.update(spi_hz=a.spi_hz,spi_write_ms=stats([r['spi_write_ms'] for r in passed]),spi_read_ms=stats([r['spi_read_ms'] for r in passed]))
    json.dump(dict(summary=summary,frames=rows),open(a.out/'results.json','w'),indent=1)
    print(json.dumps(summary,indent=1))
    print(f"\n{len(passed)}/{len(rows)-summary['skipped_empty']} sent frames bit-exact"+(f", {len(bad)} FAILED" if bad else '')+f"  ->  {a.out/'results.json'}")
    sys.exit(1 if bad else 0)

if __name__=='__main__':main()
