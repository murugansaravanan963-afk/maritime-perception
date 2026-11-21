#!/usr/bin/env python3
"""Exhaustive hash-neighbour hints for annotation-reviewed candidates.

No cap, no automatic duplicate decisions, split assignment, or review approval.
Capture metadata and actual image inspection remain necessary.
"""
import argparse,json,hashlib,sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from collections import Counter
import numpy as np
from PIL import Image
from scipy.fft import dctn
ROOT=Path(__file__).resolve().parents[1]

def sha(data):return hashlib.sha256(data).hexdigest()

def phash(im):
    a=np.asarray(im.convert('L').resize((32,32),Image.Resampling.LANCZOS),dtype=float)
    low=dctn(a,type=2,norm='ortho')[:8,:8].flatten()
    threshold=float(np.median(low[1:]))
    bits=low>threshold
    bits[0]=False
    return sum(int(b)<<i for i,b in enumerate(bits))

def neighbour_pairs(features,distance_d=4,distance_p=8):
    """Visit every unordered pair; absence of a hint is not proof of independence."""
    for i,a in enumerate(features):
        for b in features[i+1:]:
            dh=(int(a['dhash64'],16)^int(b['dhash64'],16)).bit_count()
            ph=(int(a['phash64'],16)^int(b['phash64'],16)).bit_count()
            if dh<=distance_d or ph<=distance_p:
                yield {'a':a['id'],'b':b['id'],'dhash_distance':dh,'phash_distance':ph,
                       'guard_a':a['split_guard_group'],'guard_b':b['split_guard_group'],
                       'cross_guard':a['split_guard_group']!=b['split_guard_group'],
                       'exact_pixels':a['pixel_sha256']==b['pixel_sha256']}

def verified_feature(r):
    p=Path(r['image']['path'])
    if sha(p.read_bytes())!=r['image']['sha256']:raise ValueError('Image changed: '+r['id'])
    with Image.open(p) as im:
        im=im.convert('RGB')
        if sha(str(im.size).encode()+im.tobytes())!=r['image']['pixel_sha256']:raise ValueError('Decoded pixels changed: '+r['id'])
        h=phash(im)
    return {'id':r['id'],'image_sha256':r['image']['sha256'],'pixel_sha256':r['image']['pixel_sha256'],
            'dhash64':r['image']['dhash64'],'phash64':format(h,'016x'),'split_guard_group':r['split_guard_group']}

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--manifest',type=Path,default=ROOT/'staging/annotation_review_pass.jsonl')
    ap.add_argument('--out',type=Path,default=ROOT/'reports/reviewed_neighbours_001');args=ap.parse_args()
    if args.out.exists():raise FileExistsError('Use a new immutable audit output folder')
    data=args.manifest.read_bytes();rows=[json.loads(x) for x in data.decode('utf8').splitlines() if x.strip()]
    assert len({r['id'] for r in rows})==len(rows)
    features=[]
    with ThreadPoolExecutor(max_workers=4) as pool:
        for f in pool.map(verified_feature,rows):
            features.append(f)
            if len(features)%200==0:print('Verified hash features',len(features),'/',len(rows),flush=True)
    pairs=list(neighbour_pairs(features))
    report={'state':'EXHAUSTIVE_HASH_HINTS_NOT_DUPLICATE_CERTIFICATION','manifest':str(args.manifest),
            'manifest_sha256':sha(data),'images':len(rows),'unordered_pairs_compared':len(rows)*(len(rows)-1)//2,
            'dhash_threshold':4,'phash_threshold':8,'selection':'dHash<=4 OR pHash<=8; no per-image or per-group cap',
            'hash_definition':'pHash: grayscale32x32 Lanczos; orthonormal DCT-II; top-left8x8; median of63 AC; DC bit0.',
            'hint_pairs':len(pairs),'cross_guard_hint_pairs':sum(p['cross_guard'] for p in pairs),
            'exact_pixel_hint_pairs':sum(p['exact_pixels'] for p in pairs),
            'manual_pair_review_completed':False,'duplicates_removed':0,'group_changes':0,'release_approvals':0,
            'limitations':['Whole-frame perceptual hashes can miss crops, overlays, large scale changes and different views of one event.',
                          'Similar water backgrounds can flag distinct scenes. Hints require actual image/metadata adjudication.',
                          'Source videos, sites and photo author/event metadata must also be kept together. No independence certificate is created.']}
    args.out.mkdir(parents=True)
    for name,rs in [('features.jsonl',features),('pairs.jsonl',pairs)]:
        (args.out/name).write_text(''.join(json.dumps(x)+'\n' for x in rs),encoding='utf8')
    (args.out/'summary.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf8')
    print(json.dumps(report,indent=2))
if __name__=='__main__':main()
