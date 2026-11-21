"""Make our own test set for the water-segmentation and horizon models.

LaRS keeps its test masks private, so we hold out whole recordings from LaRS train
(20% of the images of each source except MaSTr) as `test`. LaRS val stays `val`.
LaRS test images (no masks) become `lars_test`.

Recording = YouTube video (yt###) or sequence (davimar_seq_01, inhouse_seq_03, ...).
Recordings with any pair of near-identical frames (thumbnail correlation >= 0.93) are joined
into one session; whole sessions are held out. MaSTr1325 stays train-only: its ids are not in time order, so the same foggy day shows up
under ids hundreds apart and no id-based grouping separates trips.
Afterwards every test image is compared with every train image (32x32 grey thumbnails,
correlation); train images that look practically identical to a test image are dropped.

  python scene/split_scene_data.py --data scene/data/scene_lars_v1
"""
import argparse, json, random, re, collections
from pathlib import Path
import cv2, numpy as np


def recording(i):
    clip = i.rsplit('_', 1)[0]
    if clip.startswith('yt'):
        return clip.split('_')[0]
    m = re.match(r'(mastr_(?:old_)?)(\d+)$', clip)
    if m:
        return f'{m.group(1)}block{int(m.group(2)) // 50}'
    m = re.match(r'(orca_usv_inland_N\d+)_\d+$', clip)
    return m.group(1) if m else clip


TRAIN_ONLY_SOURCES = {'mastr'}


def source(i):
    return re.split(r'[_\d]', i)[0]


def thumb(path):
    t = cv2.resize(cv2.imread(str(path), cv2.IMREAD_GRAYSCALE), (32, 32), interpolation=cv2.INTER_AREA).astype(np.float32).ravel()
    t -= t.mean()
    return t / (np.linalg.norm(t) + 1e-6)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--holdout', type=float, default=0.20)
    ap.add_argument('--seed', type=int, default=20261101)
    ap.add_argument('--session-corr', type=float, default=0.93)
    ap.add_argument('--dup-corr', type=float, default=0.93)
    a = ap.parse_args()
    path = Path(a.data) / 'scene_labels.json'
    meta = json.load(open(path))
    recs = meta['records']
    for r in recs:
        r.setdefault('lars_split', r['split'])
        r['split'] = {'train': 'train', 'val': 'val', 'test': 'lars_test'}[r['lars_split']]
        r['recording'] = recording(r['id'])
        r.pop('dropped', None); r.pop('session', None)

    # sessions: recordings joined when any of their frames look nearly identical
    # (sequence names cannot be trusted: davimar_seq_16 and _17 are the same place and day)
    root = Path(a.data)
    train = [r for r in recs if r['split'] == 'train']
    X = np.stack([thumb(root / r['image']) for r in train])
    names = sorted({r['recording'] for r in train})
    ix = {k: i for i, k in enumerate(names)}
    parent = list(range(len(names)))

    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    g = np.array([ix[r['recording']] for r in train])
    I, J = np.nonzero(np.triu(X @ X.T >= a.session_corr, 1))
    for i, j in zip(I, J):
        pi, pj = find(g[i]), find(g[j])
        if pi != pj:
            parent[pi] = pj
    sessions = collections.defaultdict(list)
    for r in train:
        r['session'] = f'session{find(ix[r["recording"]])}'
        sessions[r['session']].append(r)

    # hold out whole sessions, 20% of each source; sessions touching MaSTr stay in train
    rng = random.Random(a.seed)
    by_src = collections.defaultdict(list)
    for sid, rs in sorted(sessions.items()):
        if any(source(r['id']) in TRAIN_ONLY_SOURCES for r in rs):
            continue
        main_src = collections.Counter(source(r['id']) for r in rs).most_common(1)[0][0]
        by_src[main_src].append(sid)
    for src in sorted(by_src):
        groups = by_src[src]
        rng.shuffle(groups)
        total = sum(len(sessions[s]) for s in groups)
        taken = 0
        for sid in groups:
            if taken >= a.holdout * total:
                break
            if len(sessions[sid]) > 0.5 * a.holdout * total + 5:   # skip a session that alone overshoots
                continue
            for r in sessions[sid]:
                r['split'] = 'test'
            taken += len(sessions[sid])

    # near-duplicate guard: drop train images practically identical to a test image
    te = [r for r in recs if r['split'] == 'test']
    tr = [r for r in recs if r['split'] == 'train']
    T = np.stack([thumb(root / r['image']) for r in te])
    R = np.stack([thumb(root / r['image']) for r in tr])
    corr = R @ T.T
    best = corr.max(1)
    dropped = 0
    for r, c in zip(tr, best):
        if c >= a.dup_corr:
            r['split'] = 'dropped'; r['dropped'] = f'near-duplicate of a test image (corr {c:.3f})'; dropped += 1

    meta['split_policy'] = dict(holdout=a.holdout, seed=a.seed, session_corr=a.session_corr, dup_corr=a.dup_corr,
                                rule='whole sessions (recordings joined by near-identical frames) from LaRS train '
                                     'held out as test; MaSTr train-only; LaRS val = val')
    json.dump(meta, open(path, 'w'), indent=0)
    summ = {}
    for s in ['train', 'val', 'test', 'dropped', 'lars_test']:
        rs = [r for r in recs if r['split'] == s]
        summ[s] = dict(images=len(rs), recordings=len({r['recording'] for r in rs}),
                       sessions=len({r.get('session') for r in rs}),
                       masks=sum(r['mask'] is not None for r in rs),
                       horizon_line_ok=sum(bool(r['horizon'] and r['horizon'].get('line_ok')) for r in rs),
                       sea_like=sum(r['scene_type'] == 'sea_like' for r in rs),
                       by_source=dict(collections.Counter(source(r['id']) for r in rs)))
    shared = {r['recording'] for r in recs if r['split'] == 'train'} & {r['recording'] for r in recs if r['split'] == 'test'}
    shared_s = {r['session'] for r in recs if r['split'] == 'train'} & {r['session'] for r in recs if r['split'] == 'test'}
    summ['check'] = dict(recordings_in_train_and_test=len(shared), sessions_in_train_and_test=len(shared_s), dropped_near_duplicates=dropped,
                         max_test_train_corr_after=float(max([c for r, c in zip(tr, best) if r['split'] == 'train'], default=0)))
    json.dump(summ, open(Path(a.data) / 'split_summary.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
