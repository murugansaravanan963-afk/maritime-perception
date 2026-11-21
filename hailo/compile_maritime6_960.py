#!/usr/bin/env python3
"""Compile maritime6_960.onnx (YOLOv8s, 6 classes, 960x960) to a Hailo-8L HEF with RAW outputs.

Run inside the Hailo Dataflow Compiler (DFC 3.33 / Model Zoo 2.17 environment) on x86 Linux or
WSL2, from this folder:
    python compile_maritime6_960.py --onnx hailo_export/maritime6_960.onnx --calib hailo_export/calib

Output contract (the Pi scripts and the RTL team rely on it; same layout as the power-line HEF):
  * input : 960x960x3 uint8 RGB, letterboxed like Ultralytics (gray 114 padding). /255 runs on chip,
            so the Pi feeds 0-255 values and must NOT divide by 255.
  * output "bbox"  : 18,900 anchors x 64  (DFL logits, sides l,t,r,b x 16 bins)  -- one quant scale
  * output "score" : 18,900 anchors x 6   (class probabilities, sigmoid applied on chip)
    Anchor order: stride 8 grid (120x120), then 16 (60x60), then 32 (30x30), row-major.
  * No NMS on chip. DFL, decode, threshold and NMS stay on the Pi CPU (or the FPGA block).

--layout split6 is a fallback (the standard Model Zoo cut: 3 box + 3 class tensors). It changes
the RTL contract (one quant scale per stride), so use it only if the combined layout will not parse.
"""
import argparse,json,hashlib
from pathlib import Path
import numpy as np
import cv2

SIZE=960
COMBINED_TENSORS=['/model.22/Split_output_0','/model.22/Sigmoid_output_0']   # [1,64,18900], [1,6,18900]
COMBINED_NODES=['/model.22/Split','/model.22/Sigmoid']
SPLIT6_NODES=[f'/model.22/{b}.{i}/{b}.{i}.2/Conv' for i in range(3) for b in ('cv2','cv3')]

def letterbox(img,size=SIZE):
    """Ultralytics letterbox: keep aspect, resize, centre-pad with 114. Returns RGB uint8."""
    h,w=img.shape[:2];r=min(size/h,size/w);nw,nh=round(w*r),round(h*r)
    img=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LINEAR)
    top=(size-nh)//2;left=(size-nw)//2
    out=np.full((size,size,3),114,np.uint8);out[top:top+nh,left:left+nw]=img
    return cv2.cvtColor(out,cv2.COLOR_BGR2RGB)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--onnx',type=Path,required=True);ap.add_argument('--calib',type=Path,required=True)
    ap.add_argument('--calib-count',type=int,default=1024);ap.add_argument('--name',default='maritime6_960')
    ap.add_argument('--layout',choices=['combined','split6'],default='combined')
    ap.add_argument('--opt-level',type=int,default=2,help='2 is best with a GPU; use 1 on a CPU-only machine')
    a=ap.parse_args()
    onnx_path=a.onnx
    if a.layout=='combined':
        # Cut the graph so its only outputs are the two tensors above; DFL/decode nodes are dropped.
        import onnx.utils
        onnx_path=Path(f'{a.name}_raw.onnx')
        onnx.utils.extract_model(str(a.onnx),str(onnx_path),['images'],COMBINED_TENSORS)
        print('cut ONNX ->',onnx_path,flush=True)
    from hailo_sdk_client import ClientRunner
    runner=ClientRunner(hw_arch='hailo8l')
    runner.translate_onnx_model(str(onnx_path),a.name,start_node_names=['images'],
                                end_node_names=COMBINED_NODES if a.layout=='combined' else SPLIT6_NODES,
                                net_input_shapes={'images':[1,3,SIZE,SIZE]})
    runner.save_har(f'{a.name}_parsed.har')
    hn=runner.get_hn();hn=json.loads(hn) if isinstance(hn,str) else hn
    outs=hn['net_params']['output_layers_order']
    shapes={n:hn['layers'][n]['output_shapes'][0] for n in outs}
    print('parsed outputs:',json.dumps(shapes),flush=True)
    lines=['normalization1 = normalization([0.0, 0.0, 0.0], [255.0, 255.0, 255.0])']
    if a.layout=='split6':
        # Model Zoo recipe: sigmoid moved onto the three class convolutions.
        lines+=[f'change_output_activation({n}, sigmoid)' for n in outs if shapes[n][-1]==6]
    elif len(outs)!=2:
        raise SystemExit(f'Expected 2 outputs, parser produced {shapes}. Paste this message back.')
    lines.append(f'model_optimization_flavor(optimization_level={a.opt_level}, compression_level=0)')
    script='\n'.join(lines)+'\n';Path(f'{a.name}.alls').write_text(script);runner.load_model_script(script)
    files=sorted(p for p in a.calib.iterdir() if p.suffix.lower() in ('.jpg','.jpeg','.png'))[:a.calib_count]
    calib=np.stack([letterbox(cv2.imread(str(p))) for p in files])
    print('calibration',calib.shape,calib.dtype,flush=True)
    runner.optimize(calib);runner.save_har(f'{a.name}_quantized.har')
    hef=runner.compile();Path(f'{a.name}.hef').write_bytes(hef)
    report={'hef':f'{a.name}.hef','hef_sha256':hashlib.sha256(hef).hexdigest(),'onnx_sha256':hashlib.sha256(a.onnx.read_bytes()).hexdigest(),
            'layout':a.layout,'output_layers':outs,'output_shapes':shapes,'alls':script,'calibration_images':len(files),
            'input':'960x960x3 uint8 RGB letterbox(114); normalization on chip'}
    Path('compile_report.json').write_text(json.dumps(report,indent=2)+'\n');print(json.dumps(report,indent=2))
if __name__=='__main__':main()
