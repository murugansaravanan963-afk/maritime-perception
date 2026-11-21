#!/usr/bin/env python3
"""Train standard YOLOv8 only after an actual reviewed release passes preflight."""
import argparse,json,os,re,sys,hashlib,importlib.metadata
from datetime import datetime,timezone
from pathlib import Path
from release_gate import digest,NAMES
from preflight_release import verify_release,runtime_yaml
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser();p.add_argument('--release',type=Path,required=True);p.add_argument('--weights',type=Path,required=True);p.add_argument('--name',required=True);p.add_argument('--imgsz',type=int,default=640);p.add_argument('--epochs',type=int,default=150);p.add_argument('--batch',type=int,default=8);p.add_argument('--workers',type=int,default=4);p.add_argument('--device',default='0');p.add_argument('--aug',choices=['light','standard'],default='standard',help='light = pilot_r0 settings; standard = Ultralytics default augmentation');p.add_argument('--patience',type=int,default=50);p.add_argument('--dry-run',action='store_true');a=p.parse_args()
 if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]{0,100}',a.name):raise ValueError('Use a simple unique run name')
 if a.imgsz<320 or a.imgsz%32 or a.epochs<1 or a.batch<1 or a.workers<0:raise ValueError('Invalid training dimensions/options')
 release=verify_release(a.release)
 pins=json.loads((ROOT/'configs/pretrained_weights.json').read_text());wh=digest(a.weights.read_bytes())
 allowed={v['sha256']:v for v in pins['weights']}
 if wh not in allowed:raise ValueError('Initialization must match a pinned original COCO YOLOv8 checkpoint; old maritime runs are not allowed')
 if importlib.metadata.version('ultralytics')!=pins['training_ultralytics_version']:raise ValueError('Ultralytics differs from the training environment pin')
 run=ROOT/'runs/detect'/a.name
 if run.exists():raise FileExistsError('Run already exists; choose a fresh name')
 data=runtime_yaml(a.release,ROOT/'runtime'/(a.name+'_dataset.yaml'))
 cfg={'data':str(data),'epochs':a.epochs,'imgsz':a.imgsz,'batch':a.batch,'workers':a.workers,'device':a.device,'project':str(ROOT/'runs/detect'),'name':a.name,'exist_ok':False,'seed':42,'deterministic':True,'optimizer':'AdamW','lr0':.001,'patience':a.patience,'pretrained':True,'cache':False,'amp':True,'plots':True,'save':True,'save_period':10,'mixup':0.0,'copy_paste':0.0,'fliplr':.5,'flipud':0.0,'degrees':0.0,'close_mosaic':10}
 # pilot_r0 (val mAP50 0.604) used light mosaic/scale; standard restores Ultralytics defaults,
 # which help small, scale-varying targets such as distant buoys and swimmers.
 cfg.update({'mosaic':.25,'scale':.15,'translate':.05} if a.aug=='light' else {'mosaic':1.0,'scale':.5,'translate':.1})
 receipt={'created_utc':datetime.now(timezone.utc).isoformat(),'release':release,'initial_weights_sha256':wh,'initial_model':allowed[wh]['model'],'ultralytics_version':importlib.metadata.version('ultralytics'),'config':cfg,'test_used_for_model_selection':False,'target_map50':.9,'target_is_guarantee':False}
 if a.dry_run:print(json.dumps(receipt,indent=2));return
 os.environ.setdefault('YOLO_CONFIG_DIR',str(ROOT/'.cache/ultralytics'));(Path(os.environ['YOLO_CONFIG_DIR'])/'Ultralytics').mkdir(parents=True,exist_ok=True);os.environ.setdefault('WANDB_MODE','disabled');os.environ.setdefault('MPLBACKEND','Agg')
 import torch
 from ultralytics import YOLO,settings
 if not torch.cuda.is_available():raise RuntimeError('CUDA required for long training; use a Sol GPU allocation')
 settings.update({'sync':False,'wandb':False,'comet':False,'mlflow':False,'clearml':False,'neptune':False,'dvc':False,'hub':False})
 # Remove generated label caches so training reads the verified text labels.
 for split in ['train','val']:
  cache=a.release.resolve()/'labels'/(split+'.cache')
  if cache.is_file():cache.unlink()
 model=YOLO(str(a.weights.resolve()))
 if len(model.names)!=80 or model.task!='detect':raise ValueError('Expected original 80-class COCO detector initialization')
 receipt['torch_version']=torch.__version__;receipt['cuda_version']=torch.version.cuda;receipt['gpu']=torch.cuda.get_device_name(0)
 receipt['packages']={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')}
 out=ROOT/'reports/training';out.mkdir(parents=True,exist_ok=True);rp=out/(a.name+'.json')
 if rp.exists():raise FileExistsError('Training receipt exists; select another run name')
 rp.write_text(json.dumps(receipt,indent=2)+'\n',encoding='utf8')
 model.train(**cfg)
 # Use validation only. A final sealed-test run is a separate selection step.
 best=run/'weights/best.pt';m=YOLO(str(best));val=m.val(data=str(data),split='val',imgsz=a.imgsz,batch=a.batch,device=a.device,conf=.001,iou=.7,plots=True,project=str(ROOT/'runs/validation'),name=a.name,exist_ok=False)
 indices=list(map(int,val.box.ap_class_index));per={NAMES[c]:{'AP50':None,'AP50_95':None} for c in range(6)}
 for idx,c in enumerate(indices):per[NAMES[c]]={'AP50':float(val.box.all_ap[idx,0]),'AP50_95':float(val.box.all_ap[idx].mean())}
 result={'weights':str(best),'weights_sha256':digest(best.read_bytes()),'split':'val','mAP50':float(val.box.map50),'mAP50_95':float(val.box.map),'per_class':per,'all_six_classes_present':set(indices)==set(range(6)),'release_fingerprint_sha256':release['release_fingerprint_sha256'],'test_evaluated':False}
 (out/(a.name+'_validation.json')).write_text(json.dumps(result,indent=2)+'\n',encoding='utf8');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
