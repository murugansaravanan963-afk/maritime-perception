"""Build the LaRS scene dataset for the water-segmentation, horizon and sea-state models.

Every image is letterboxed to 640x384 (keep aspect, pad 114). Masks get the same letterbox,
pad = 255 (ignore). Mask values (LaRS semantic): 0 obstacle, 1 water, 2 sky, 255 ignore.

Horizon ground truth comes from the mask: for each column the topmost water pixel is the
water edge; a robust (Huber) line fit through those points gives the horizon / water-edge line,
stored as y at x=0 and x=W-1 of the 640-wide canvas.

Sea-state labels come from LaRS image_annotations.json:
  calm      = waves == still        (rough: only 10 images in all of LaRS, counted as not calm)
  fog, glitter
  low_light = extra_dark or lighting in {dusk_like, night_like}
LaRS test has scene labels but no public masks, so test is used only by the sea-state model.

Usage:
  python scene/build_scene_data.py --lars-images D:/Projects/maritime-perception/data/lars_images \
      --lars-ann D:/Projects/maritime-perception/data/lars_v1.0.0_annotations --out scene_lars_v1
"""
import argparse, json, hashlib
from pathlib import Path
import cv2, numpy as np

W, H = 640, 384
PAD_IMG, IGNORE = 114, 255
STATE_KEYS = ['calm', 'fog', 'glitter', 'low_light']


def letterbox_params(w, h):
    r = min(W / w, H / h)
    nw, nh = round(w * r), round(h * r)
    return r, (W - nw) // 2, (H - nh) // 2, nw, nh


def letterbox(arr, interp, pad):
    h, w = arr.shape[:2]
    r, px, py, nw, nh = letterbox_params(w, h)
    out = np.full((H, W) + arr.shape[2:], pad, arr.dtype)
    out[py:py + nh, px:px + nw] = cv2.resize(arr, (nw, nh), interpolation=interp)
    return out, (r, px, py)


def water_edge(mask, things=None, look=6):
    """Per column: topmost water row, or -1 if the column has no water.
    If `things` (boat/buoy/swimmer pixels) is given, columns where a thing sits directly on
    top of the water edge (within `look` rows) are set to -1: the true edge is hidden there."""
    water = mask == 1
    has = water.any(0)
    top = np.where(has, water.argmax(0), -1)
    if things is not None:
        for x in np.flatnonzero(top > 0):
            t = top[x]
            if things[max(0, t - look):t, x].any():
                top[x] = -1
    return top


def thing_mask(png, segments):
    """Pixels of LaRS 'thing' segments (category id >= 11: boats, buoys, swimmers, ...)."""
    rgb = cv2.imread(str(png), cv2.IMREAD_COLOR)[..., ::-1].astype(np.int64)
    ids = rgb[..., 0] + 256 * rgb[..., 1] + 65536 * rgb[..., 2]
    keep = [sg['id'] for sg in segments if sg['category_id'] >= 11]
    return np.isin(ids, keep).astype(np.uint8)


def fit_line(top, x0, x1, iters=8):
    """Huber IRLS line fit through (x, top[x]) for columns x0..x1-1 with water."""
    xs = np.arange(len(top))
    ok = (top >= 0) & (xs >= x0) & (xs < x1)
    if ok.sum() < 0.3 * (x1 - x0):
        return None
    x, y = xs[ok].astype(np.float64), top[ok].astype(np.float64)
    w = np.ones_like(x)
    for _ in range(iters):
        a, b = np.polyfit(x, y, 1, w=np.sqrt(w))
        res = y - (a * x + b)
        s = max(1.4826 * np.median(np.abs(res - np.median(res))), 1.0)
        k = 1.345 * s
        w = np.where(np.abs(res) <= k, 1.0, k / np.abs(res))
    res = y - (a * x + b)
    inlier = float(np.mean(np.abs(res) <= 8))
    # line_ok: the water edge really is a line (sea / open lake). Narrow canals with V-shaped
    # banks are not; the horizon model trains on line_ok only, the water mask covers the rest.
    return dict(a=float(a), b=float(b), mad=float(np.median(np.abs(res))), inlier=inlier,
                cover=float(ok.sum() / (x1 - x0)), line_ok=bool(inlier >= 0.5))


def state_labels(lab):
    sp = lab.get('special') or {}
    return dict(calm=int(lab.get('waves') == 'still'),
                fog=int(bool(sp.get('fog'))),
                glitter=int(bool(sp.get('glitter'))),
                low_light=int(bool(sp.get('extra_dark')) or lab.get('lighting') in ('dusk_like', 'night_like')),
                waves=lab.get('waves'), scene_type=lab.get('scene_type'))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--lars-images', required=True)
    ap.add_argument('--lars-ann', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--limit', type=int, default=0, help='images per split (testing only)')
    a = ap.parse_args()
    out = Path(a.out)
    records = []
    for split in ['train', 'val', 'test']:
        ann = json.load(open(Path(a.lars_ann) / split / 'image_annotations.json'))['annotations']
        pan_file = Path(a.lars_ann) / split / 'panoptic_annotations.json'
        pan = {Path(x['file_name']).stem: x['segments_info']
               for x in json.load(open(pan_file))['annotations']} if pan_file.exists() else {}
        if a.limit:
            ann = ann[:a.limit]
        (out / 'images' / split).mkdir(parents=True, exist_ok=True)
        (out / 'masks' / split).mkdir(parents=True, exist_ok=True)
        for item in ann:
            stem = Path(item['file_name']).stem
            src = Path(a.lars_images) / split / 'images' / item['file_name']
            img = cv2.imread(str(src), cv2.IMREAD_COLOR)
            if img is None:
                raise SystemExit(f'cannot read {src}')
            h0, w0 = img.shape[:2]
            lb, (r, px, py) = letterbox(img, cv2.INTER_AREA, PAD_IMG)
            cv2.imwrite(str(out / 'images' / split / f'{stem}.jpg'), lb, [cv2.IMWRITE_JPEG_QUALITY, 92])
            rec = dict(id=stem, split=split, src_w=w0, src_h=h0, scale=r, pad_x=px, pad_y=py,
                       image=f'images/{split}/{stem}.jpg', mask=None, horizon=None,
                       src_sha1=hashlib.sha1(src.read_bytes()).hexdigest())
            rec.update(state_labels(item['labels']))
            mpath = Path(a.lars_ann) / split / 'semantic_masks' / f'{stem}.png'
            if mpath.exists():
                m = cv2.imread(str(mpath), cv2.IMREAD_UNCHANGED)
                assert m.shape[:2] == (h0, w0), (stem, m.shape, img.shape)
                mlb, _ = letterbox(m, cv2.INTER_NEAREST, IGNORE)
                cv2.imwrite(str(out / 'masks' / split / f'{stem}.png'), mlb)
                rec['mask'] = f'masks/{split}/{stem}.png'
                things = None
                if stem in pan:
                    tm = thing_mask(Path(a.lars_ann) / split / 'panoptic_masks' / f'{stem}.png', pan[stem])
                    things, _ = letterbox(tm, cv2.INTER_NEAREST, 0)
                top = water_edge(mlb, things)
                nw = round(w0 * r)
                fit = fit_line(top, px, px + nw)
                if fit:
                    fit['y_left'] = fit['b']
                    fit['y_right'] = fit['a'] * (W - 1) + fit['b']
                    rec['horizon'] = fit
                # 32-point water-edge profile (row, or -1), for gate experiments
                cols = np.linspace(px, px + nw - 1, 32).round().astype(int)
                rec['edge_profile'] = top[cols].tolist()
            records.append(rec)
        print(split, sum(r['split'] == split for r in records))
    meta = dict(schema='scene-lars-1', canvas=[W, H], pad_image=PAD_IMG, mask_classes=['obstacle', 'water', 'sky'],
                ignore=IGNORE, state_keys=STATE_KEYS, records=records)
    json.dump(meta, open(out / 'scene_labels.json', 'w'), indent=0)
    # summary
    summ = {}
    for split in ['train', 'val', 'test']:
        rs = [r for r in records if r['split'] == split]
        summ[split] = dict(images=len(rs), masks=sum(r['mask'] is not None for r in rs),
                           horizon=sum(r['horizon'] is not None for r in rs),
                           horizon_line_ok=sum(bool(r['horizon'] and r['horizon']['line_ok']) for r in rs),
                           **{k: sum(r[k] for r in rs) for k in STATE_KEYS},
                           rough=sum(r['waves'] == 'rough' for r in rs))
    json.dump(summ, open(out / 'build_summary_lars_splits.json', 'w'), indent=1)
    print(json.dumps(summ, indent=1))


if __name__ == '__main__':
    main()
