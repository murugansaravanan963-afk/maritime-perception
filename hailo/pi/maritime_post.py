#!/usr/bin/env python3
"""Shared numpy code for the maritime6 960 model: letterbox, decode (DFL + boxes), NMS, mAP.

Used unchanged on the laptop (ONNX reference) and on the Raspberry Pi (HEF), so any accuracy
difference between the two comes from Hailo quantization, not from different post-processing.
"""
import numpy as np
import cv2

NAMES=['powered_vessel','sailboat','small_craft','navigation_marker','person_in_water','other_obstacle']
SIZE=960;STRIDES=(8,16,32);N_ANCHORS=sum((SIZE//s)**2 for s in STRIDES)   # 18,900

def letterbox(img,size=SIZE):
    """Ultralytics letterbox (keep aspect, centre pad 114). Returns RGB uint8, scale r, pad (left, top)."""
    h,w=img.shape[:2];r=min(size/h,size/w);nw,nh=round(w*r),round(h*r)
    small=cv2.resize(img,(nw,nh),interpolation=cv2.INTER_LINEAR)
    top=(size-nh)//2;left=(size-nw)//2
    out=np.full((size,size,3),114,np.uint8);out[top:top+nh,left:left+nw]=small
    return cv2.cvtColor(out,cv2.COLOR_BGR2RGB),r,(left,top)

def anchors():
    """(N,3) centre x, centre y, stride in input pixels; stride 8 grid first, row-major."""
    a=[]
    for s in STRIDES:
        g=SIZE//s;ys,xs=np.mgrid[0:g,0:g];a.append(np.stack([(xs.ravel()+.5)*s,(ys.ravel()+.5)*s,np.full(g*g,s)],1))
    return np.concatenate(a).astype(np.float32)
ANCHORS=anchors()

def to_anchor_major(t,channels):
    """Any (1,C,N) / (1,N,C) / (N,C) / (1,1,N,C)... tensor -> (N, channels) float32."""
    t=np.asarray(t,np.float32).reshape(-1)
    if t.size!=N_ANCHORS*channels:raise ValueError(f'expected {N_ANCHORS}x{channels} values, got {t.size}')
    return t.reshape(N_ANCHORS,channels)

def channel_major_to_anchor_major(t,channels):
    """ONNX layout (1,C,N) -> (N,C)."""
    return np.asarray(t,np.float32).reshape(channels,N_ANCHORS).T

def decode(bbox,score,conf,iou,max_det=300):
    """bbox (N,64) DFL logits, score (N,6) probabilities -> boxes xyxy in 960 input pixels, scores, classes."""
    sc=score.max(1);keep=sc>conf
    if not keep.any():return np.zeros((0,4),np.float32),np.zeros(0,np.float32),np.zeros(0,int)
    b=bbox[keep].reshape(-1,4,16);b=np.exp(b-b.max(2,keepdims=True));b/=b.sum(2,keepdims=True)
    off=(b*np.arange(16,dtype=np.float32)).sum(2)
    a=ANCHORS[keep];cx,cy,st=a[:,0],a[:,1],a[:,2]
    boxes=np.stack([cx-off[:,0]*st,cy-off[:,1]*st,cx+off[:,2]*st,cy+off[:,3]*st],1).clip(0,SIZE)
    cls=score[keep].argmax(1);sc=sc[keep]
    k=nms(boxes,sc,cls,iou)[:max_det]
    return boxes[k],sc[k],cls[k]

def nms(boxes,scores,cls,iou):
    """Class-aware greedy NMS (offset trick), highest score first."""
    off=boxes+cls[:,None]*(SIZE*4);order=scores.argsort()[::-1];keep=[]
    area=(off[:,2]-off[:,0])*(off[:,3]-off[:,1])
    while order.size:
        i=order[0];keep.append(i)
        if order.size==1:break
        r=order[1:];xx1=np.maximum(off[i,0],off[r,0]);yy1=np.maximum(off[i,1],off[r,1])
        xx2=np.minimum(off[i,2],off[r,2]);yy2=np.minimum(off[i,3],off[r,3])
        inter=np.clip(xx2-xx1,0,None)*np.clip(yy2-yy1,0,None)
        order=r[inter/(area[i]+area[r]-inter+1e-9)<iou]
    return np.array(keep,int)

def unletterbox(boxes,r,pad,w,h):
    b=boxes.copy();b[:,[0,2]]=((b[:,[0,2]]-pad[0])/r).clip(0,w);b[:,[1,3]]=((b[:,[1,3]]-pad[1])/r).clip(0,h);return b

def read_labels(path,w,h):
    """YOLO txt -> list of (cls, x1, y1, x2, y2) in image pixels."""
    out=[]
    try:lines=open(path).read().split('\n')
    except FileNotFoundError:return out
    for l in lines:
        f=l.split()
        if len(f)==5:
            c,cx,cy,bw,bh=int(f[0]),*map(float,f[1:]);out.append((c,(cx-bw/2)*w,(cy-bh/2)*h,(cx+bw/2)*w,(cy+bh/2)*h))
    return out

def _iou(a,b):
    if not len(b):return np.zeros(0)
    iw=np.clip(np.minimum(a[2],b[:,2])-np.maximum(a[0],b[:,0]),0,None);ih=np.clip(np.minimum(a[3],b[:,3])-np.maximum(a[1],b[:,1]),0,None)
    inter=iw*ih;return inter/((a[2]-a[0])*(a[3]-a[1])+(b[:,2]-b[:,0])*(b[:,3]-b[:,1])-inter)

def _ap(tp,conf,n):
    if n==0:return None
    if not len(tp):return 0.0
    o=np.argsort(-np.asarray(conf),kind='stable');tp=np.asarray(tp,float)[o]
    c=np.cumsum(tp);rec=c/n;prec=c/np.arange(1,len(tp)+1)
    prec=np.maximum.accumulate(prec[::-1])[::-1]
    return float(np.mean([prec[np.searchsorted(rec,x,side='left')] if x<=rec[-1] else 0.0 for x in np.linspace(0,1,101)]))

def evaluate(gts,preds):
    """gts {key: [(c,x1,y1,x2,y2)]}, preds {key: [(c,conf,x1,y1,x2,y2)]} -> COCO 101-point mAP50 and mAP50-95."""
    ths=np.linspace(.5,.95,10);per={}
    for c in range(len(NAMES)):
        aps=[]
        for t in ths:
            tps=[];confs=[];n=0
            for k,g in gts.items():
                gb=np.array([x[1:] for x in g if x[0]==c],float).reshape(-1,4);n+=len(gb);used=np.zeros(len(gb),bool)
                for p in sorted((p for p in preds.get(k,[]) if p[0]==c),key=lambda p:-p[1]):
                    ov=_iou(np.array(p[2:],float),gb);ov[used]=-1;j=int(ov.argmax()) if len(ov) else -1
                    if j>=0 and ov[j]>=t:used[j]=True;tps.append(1)
                    else:tps.append(0)
                    confs.append(p[1])
            aps.append(_ap(tps,confs,n))
        per[NAMES[c]]=None if aps[0] is None else {'AP50':aps[0],'AP50_95':float(np.mean(aps))}
    v=[x for x in per.values() if x]
    return {'mAP50':float(np.mean([x['AP50'] for x in v])),'mAP50_95':float(np.mean([x['AP50_95'] for x in v])),'per_class':per}
