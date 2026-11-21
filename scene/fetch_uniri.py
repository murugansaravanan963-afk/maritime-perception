"""Download only the real ship-camera RGB images of UNIRI-SeaState (CC BY 4.0) and build the
sea-state dataset: calm / moderate / rough water patches.

Source: Paulin, Pobar, Ivasic-Kos, Vorkapic, UNIRI-SeaState, Zenodo 10.5281/zenodo.19592919
(paper: IEEE OJ-ITS vol. 7, 2026, doi 10.1109/OJITS.2025.3650057).
Only `train splits/rgb/R750` (real, 5,992 images) and `test/testset-rgb` (2,022 images) are read,
with HTTP range requests (~1.3 GB instead of the 6.8 GB zip). Zip CRCs are checked on read.

Classes (Beaufort):  calm = 1-2,  moderate = 3-4,  rough = 5-8
Splits: UNIRI's own test set = test (its 90 videos share none with train).
        val = about 15% of the training videos per class (whole videos), the rest = train.

  python scene/fetch_uniri.py --out scene/data/uniri_seastate
"""
import argparse, collections, json, random, re, sys, urllib.request, zipfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from inspect_remote_zip import HTTPRangeFile

URL = 'https://zenodo.org/records/19592919/files/UNIRI-SeaState.zip?download=1'
ZIP_MD5 = '549c23ca968de97f3df8be9193c67403'
GROUPS = {1: 'calm', 2: 'calm', 3: 'moderate', 4: 'moderate', 5: 'rough', 6: 'rough', 7: 'rough', 8: 'rough'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', required=True)
    ap.add_argument('--val-frac', type=float, default=0.15)
    ap.add_argument('--seed', type=int, default=20261101)
    a = ap.parse_args()
    out = Path(a.out)
    head = urllib.request.urlopen(urllib.request.Request(URL, method='HEAD', headers={'User-Agent': 'MaritimeDataset/1.0'}), timeout=60)
    z = zipfile.ZipFile(HTTPRangeFile(head.geturl(), int(head.headers['Content-Length'])))
    want = [i for i in z.infolist() if not i.is_dir() and ('/train splits/rgb/R750/' in i.filename or '/test/testset-rgb/' in i.filename)]
    records = []
    for n, info in enumerate(want):
        parts = info.filename.split('/')
        bft, fname = int(parts[-2]), parts[-1]
        video, frame = re.match(r'(.+)_(\d+)\.png$', fname).groups()
        split = 'test' if '/test/' in info.filename else 'train'
        dst = out / 'images' / split / str(bft) / fname
        if not dst.exists() or dst.stat().st_size != info.file_size:
            dst.parent.mkdir(parents=True, exist_ok=True)
            dst.write_bytes(z.read(info))          # zipfile verifies the CRC-32 here
        records.append(dict(image=str(dst.relative_to(out)).replace('\\', '/'), split=split, beaufort=bft,
                            label=GROUPS[bft], video=video, frame=int(frame), bytes=info.file_size, crc32=info.CRC))
        if n % 500 == 0:
            print(f'{n}/{len(want)}', flush=True)

    # val = whole training videos, about val_frac of each class group
    rng = random.Random(a.seed)
    vids = collections.defaultdict(lambda: collections.defaultdict(int))
    for r in records:
        if r['split'] == 'train':
            vids[r['label']][r['video']] += 1
    for label, vc in sorted(vids.items()):
        order = sorted(vc); rng.shuffle(order)
        total, taken = sum(vc.values()), 0
        for v in order:
            if taken >= a.val_frac * total:
                break
            if vc[v] > 0.6 * a.val_frac * total:   # skip a video that alone would be most of val
                continue
            for r in records:
                if r['video'] == v and r['split'] == 'train':
                    r['split'] = 'val'
            taken += vc[v]

    summ = {s: dict(images=sum(r['split'] == s for r in records),
                    videos=len({r['video'] for r in records if r['split'] == s}),
                    by_label=dict(collections.Counter(r['label'] for r in records if r['split'] == s)),
                    by_beaufort=dict(sorted(collections.Counter(r['beaufort'] for r in records if r['split'] == s).items())))
            for s in ('train', 'val', 'test')}
    sv = {s: {r['video'] for r in records if r['split'] == s} for s in ('train', 'val', 'test')}
    summ['check'] = dict(videos_train_val=len(sv['train'] & sv['val']), videos_train_test=len(sv['train'] & sv['test']),
                         videos_val_test=len(sv['val'] & sv['test']))
    json.dump(dict(schema='uniri-seastate-1', source=URL, zip_md5=ZIP_MD5, license='CC BY 4.0',
                   groups=GROUPS, records=records), open(out / 'seastate_labels.json', 'w'), indent=0)
    json.dump(summ, open(out / 'summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
