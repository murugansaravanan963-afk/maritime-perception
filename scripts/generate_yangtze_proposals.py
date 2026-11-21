"""Generate LOCAL detector proposals only. Never writes training labels or approvals."""
import os
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
os.environ['YOLO_CONFIG_DIR']=str(ROOT/'.cache/ultralytics_proposals')
import json,hashlib,time
from ultralytics import YOLO
from release_gate import canonical_hash
from stage_candidates import jsonl
def main():
    out=ROOT/'reports/yangtze983_proposals_001'
    if out.exists():raise FileExistsError(out)
    out.mkdir()
    weight=ROOT/'weights/yolov8s.pt';weight_hash=hashlib.sha256(weight.read_bytes()).hexdigest()
    model=YOLO(str(weight))
    if model.names[8]!='boat':raise ValueError('Unexpected COCO class mapping')
    records=list(jsonl(ROOT/'staging/candidates/yangtze983.jsonl'))
    started=time.time();count=0;hist={}
    with (out/'predictions.jsonl').open('x',encoding='utf8') as f:
        for base in range(0,len(records),8):
            batch=records[base:base+8]
            predictions=model.predict(source=[r['image']['path'] for r in batch],imgsz=1280,conf=.08,iou=.65,classes=[8],device=0,half=True,batch=8,verbose=False,save=False)
            if len(predictions)!=len(batch):raise ValueError('Prediction count mismatch')
            for r,p in zip(batch,predictions):
                ps=[{'proposal_id':'b'+str(i),'model_class':'boat','model_class_id':int(c),'confidence':float(conf),'xyxy':list(map(float,xy))} for i,(xy,c,conf) in enumerate(zip(p.boxes.xyxy.cpu().tolist(),p.boxes.cls.cpu().tolist(),p.boxes.conf.cpu().tolist()))]
                f.write(json.dumps({'id':r['id'],'candidate_record_sha256':canonical_hash(r),'image_sha256':r['image']['sha256'],'weight_sha256':weight_hash,'proposals':ps,'status':'MODEL_PROPOSALS_ONLY_NOT_GROUND_TRUTH','full_frame_visual_review_required':True})+'\n')
                hist[len(ps)]=hist.get(len(ps),0)+1
            count+=len(batch);f.flush()
            if count%80==0:print(json.dumps({'proposed_images':count,'total':len(records),'seconds':round(time.time()-started,1)}),flush=True)
    summary={'images':count,'proposal_count_histogram':hist,'weights':str(weight),'weights_sha256':weight_hash,'seconds':time.time()-started,'automatic_approvals':0,'staging_candidates_unchanged':True}
    (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf8');print(json.dumps(summary,indent=2),flush=True)
if __name__=='__main__':main()
