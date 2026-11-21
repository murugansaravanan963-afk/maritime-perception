"""Train / evaluate / export the scene models on scene_lars_v1.

Train (fixed epochs, the last epoch is the result; val is only watched, never used to pick):
  python scene/train_scene.py --data scene/data/scene_lars_v1 --task seg --name seg_r18
Evaluate once on another split (state has LaRS test labels; seg/horizon have no public test masks):
  python scene/train_scene.py --data ... --task state --eval runs_scene/state_r18/last.pt --split test
Export ONNX for the Hailo compiler:
  python scene/train_scene.py --data ... --task seg --export runs_scene/seg_r18/last.pt
"""
import argparse, json, math, random, time
from pathlib import Path
import cv2, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from scene_models import SceneModel, ExportWrapper, TASK_HEADS, H, W, NUM_SEG

MEAN = np.array([0.485, 0.456, 0.406], np.float32) * 255
STD = np.array([0.229, 0.224, 0.225], np.float32) * 255
STATE_KEYS = ['calm', 'fog', 'glitter', 'low_light']
IGNORE = 255


def needed(task, rec):
    if task == 'seg':
        return rec['mask'] is not None
    if task == 'horizon':
        return bool(rec['horizon'] and rec['horizon']['line_ok'])
    if task == 'multi':
        return rec['mask'] is not None
    return True


class SceneData(Dataset):
    def __init__(self, root, split, task, train):
        meta = json.load(open(Path(root) / 'scene_labels.json'))
        self.root, self.task, self.train = Path(root), task, train
        self.recs = [r for r in meta['records'] if r['split'] == split and needed(task, r)]

    def __len__(self):
        return len(self.recs)

    def __getitem__(self, i):
        r = self.recs[i]
        img = cv2.imread(str(self.root / r['image']))[..., ::-1]  # RGB
        mask = cv2.imread(str(self.root / r['mask']), cv2.IMREAD_UNCHANGED) if r['mask'] else None
        hz = r['horizon']
        pts = np.array([[0, hz['y_left']], [W - 1, hz['y_right']]], np.float64) if hz else None
        if self.train:
            s = random.uniform(0.8, 1.2)
            ang = random.uniform(-5, 5)
            M = cv2.getRotationMatrix2D((W / 2, H / 2), ang, s)
            M[:, 2] += (random.uniform(-0.08, 0.08) * W, random.uniform(-0.08, 0.08) * H)
            if random.random() < 0.5:  # horizontal flip, folded into the affine
                M = np.array([[-1, 0, W - 1], [0, 1, 0]], np.float64) @ np.vstack([M, [0, 0, 1]])
            img = cv2.warpAffine(np.ascontiguousarray(img), M, (W, H), flags=cv2.INTER_LINEAR,
                                 borderValue=(114, 114, 114))
            if mask is not None:
                mask = cv2.warpAffine(mask, M, (W, H), flags=cv2.INTER_NEAREST, borderValue=IGNORE)
            if pts is not None:
                p = np.hstack([pts, np.ones((2, 1))]) @ M.T
                a = (p[1, 1] - p[0, 1]) / (p[1, 0] - p[0, 0])
                b = p[0, 1] - a * p[0, 0]
                pts = np.array([[0, b], [W - 1, a * (W - 1) + b]])
            hsv = cv2.cvtColor(np.ascontiguousarray(img), cv2.COLOR_RGB2HSV).astype(np.float32)
            hsv[..., 0] = (hsv[..., 0] + random.uniform(-5, 5)) % 180
            hsv[..., 1] *= random.uniform(0.7, 1.3)
            hsv[..., 2] *= random.uniform(0.75, 1.25)
            img = cv2.cvtColor(np.clip(hsv, 0, 255).astype(np.uint8), cv2.COLOR_HSV2RGB)
        x = torch.from_numpy(((img.astype(np.float32) - MEAN) / STD).transpose(2, 0, 1).copy())
        y = {'state': torch.tensor([r[k] for k in STATE_KEYS], dtype=torch.float32)}
        y['mask'] = torch.from_numpy(mask.astype(np.int64)) if mask is not None else torch.full((H, W), IGNORE)
        ok = pts is not None and bool(hz['line_ok'])
        y['horizon'] = torch.tensor([pts[0, 1] / H, pts[1, 1] / H] if ok else [0.0, 0.0], dtype=torch.float32)
        y['horizon_ok'] = torch.tensor(float(ok))
        return x, y


def losses(out, y, pos_weight):
    L = {}
    if 'seg' in out:
        logits = F.interpolate(out['seg'], size=(H, W), mode='bilinear', align_corners=False)
        L['seg'] = F.cross_entropy(logits, y['mask'], ignore_index=IGNORE)
    if 'horizon' in out:
        ok = y['horizon_ok']
        l = F.smooth_l1_loss(out['horizon'], y['horizon'], beta=0.02, reduction='none').mean(1)
        L['horizon'] = (l * ok).sum() / ok.sum().clamp(min=1) * 10
    if 'state' in out:
        L['state'] = F.binary_cross_entropy_with_logits(out['state'], y['state'], pos_weight=pos_weight)
    return L


def average_precision(score, label):
    o = np.argsort(-score)
    l = label[o]
    if l.sum() == 0:
        return float('nan')
    tp = np.cumsum(l)
    prec = tp / np.arange(1, len(l) + 1)
    return float((prec * l).sum() / l.sum())


@torch.no_grad()
def evaluate(model, loader, dev):
    model.eval()
    inter = np.zeros(NUM_SEG); union = np.zeros(NUM_SEG)
    hz_err, st_score, st_label = [], [], []
    for x, y in loader:
        out = model(x.to(dev))
        if 'seg' in out:
            pred = F.interpolate(out['seg'].float(), size=(H, W), mode='bilinear', align_corners=False).argmax(1).cpu()
            m = y['mask']; v = m != IGNORE
            for c in range(NUM_SEG):
                p, g = (pred == c) & v, (m == c) & v
                inter[c] += (p & g).sum().item(); union[c] += (p | g).sum().item()
        if 'horizon' in out:
            ok = y['horizon_ok'].bool()
            if ok.any():
                e = (out['horizon'].float().cpu()[ok] - y['horizon'][ok]).abs() * H   # px at 384 height
                hz_err.append(e.mean(1).numpy())
        if 'state' in out:
            st_score.append(torch.sigmoid(out['state'].float()).cpu().numpy()); st_label.append(y['state'].numpy())
    res = {}
    if union.sum():
        iou = inter / np.maximum(union, 1)
        res.update(miou=float(iou.mean()), iou_obstacle=float(iou[0]), iou_water=float(iou[1]), iou_sky=float(iou[2]))
    if hz_err:
        e = np.concatenate(hz_err)
        res.update(horizon_mae_px=float(e.mean()), horizon_median_px=float(np.median(e)),
                   horizon_within_5px=float((e <= 5).mean()), horizon_within_10px=float((e <= 10).mean()),
                   horizon_n=int(len(e)))
    if st_score:
        s, l = np.concatenate(st_score), np.concatenate(st_label)
        for j, k in enumerate(STATE_KEYS):
            pr = s[:, j] >= 0.5
            tp = (pr & (l[:, j] == 1)).sum(); fp = (pr & (l[:, j] == 0)).sum(); fn = (~pr & (l[:, j] == 1)).sum()
            tpr = tp / max(l[:, j].sum(), 1); tnr = ((~pr) & (l[:, j] == 0)).sum() / max((l[:, j] == 0).sum(), 1)
            res[f'{k}_ap'] = average_precision(s[:, j], l[:, j])
            res[f'{k}_f1'] = float(2 * tp / max(2 * tp + fp + fn, 1))
            res[f'{k}_bal_acc'] = float((tpr + tnr) / 2)
            res[f'{k}_pos'] = int(l[:, j].sum())
        res['state_map'] = float(np.nanmean([res[f'{k}_ap'] for k in STATE_KEYS]))
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', required=True)
    ap.add_argument('--task', required=True, choices=list(TASK_HEADS))
    ap.add_argument('--name', default=None)
    ap.add_argument('--project', default='runs_scene')
    ap.add_argument('--epochs', type=int, default=80)
    ap.add_argument('--batch', type=int, default=32)
    ap.add_argument('--lr', type=float, default=1e-3)
    ap.add_argument('--workers', type=int, default=4)
    ap.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
    ap.add_argument('--limit', type=int, default=0, help='images per split (smoke test only)')
    ap.add_argument('--no-pretrained', action='store_true')
    ap.add_argument('--eval', default=None, help='checkpoint to evaluate on --split, no training')
    ap.add_argument('--split', default='val')
    ap.add_argument('--export', default=None, help='checkpoint to export as ONNX')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    random.seed(a.seed); np.random.seed(a.seed); torch.manual_seed(a.seed)
    dev = torch.device(a.device)

    if a.export or a.eval:
        ck = torch.load(a.export or a.eval, map_location='cpu', weights_only=False)
        model = SceneModel(ck['task'], pretrained=False); model.load_state_dict(ck['model']); model.to(dev)
        if a.export:
            onnx = Path(a.export).with_suffix('.onnx')
            names = [h for h in ('seg', 'horizon', 'state') if h in model.heads]
            torch.onnx.export(ExportWrapper(model.eval().cpu()), torch.zeros(1, 3, H, W), str(onnx),
                              input_names=['images'], output_names=names, opset_version=11, dynamo=False)
            print('wrote', onnx, 'outputs', names)
            return
        ds = SceneData(a.data, a.split, ck['task'], False)
        res = evaluate(model, DataLoader(ds, a.batch, num_workers=a.workers), dev)
        res.update(split=a.split, images=len(ds), checkpoint=a.eval)
        Path(a.eval).with_name(f'eval_{a.split}.json').write_text(json.dumps(res, indent=1))
        print(json.dumps(res, indent=1))
        return

    run_dir = Path(a.project) / (a.name or a.task); run_dir.mkdir(parents=True, exist_ok=True)
    tr, va = SceneData(a.data, 'train', a.task, True), SceneData(a.data, 'val', a.task, False)
    if a.limit:
        tr.recs, va.recs = tr.recs[:a.limit], va.recs[:a.limit]
    print(f'{a.task}: train {len(tr)} val {len(va)}')
    lab = np.array([[r[k] for k in STATE_KEYS] for r in tr.recs], np.float32)
    pos = lab.sum(0)
    pos_weight = torch.tensor(np.clip((len(lab) - pos) / np.maximum(pos, 1), 1, 10), dtype=torch.float32, device=dev)
    tl = DataLoader(tr, a.batch, shuffle=True, num_workers=a.workers, drop_last=len(tr) > a.batch, pin_memory=True)
    vl = DataLoader(va, a.batch, num_workers=a.workers)

    model = SceneModel(a.task, pretrained=not a.no_pretrained).to(dev)
    bb = list(model.backbone.parameters()); hd = list(model.heads.parameters())
    opt = torch.optim.AdamW([{'params': bb, 'lr': a.lr * 0.3}, {'params': hd, 'lr': a.lr}], weight_decay=1e-4)
    steps = a.epochs * max(len(tl), 1); warm = min(300, steps // 10)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min(1, (i + 1) / max(warm, 1)) * 0.5 * (1 + math.cos(math.pi * i / steps)))
    amp = dev.type == 'cuda'
    scaler = torch.amp.GradScaler(enabled=amp)
    hist = []
    for ep in range(a.epochs):
        model.train(); t0 = time.time(); tot = {}
        for x, y in tl:
            x = x.to(dev, non_blocking=True); y = {k: v.to(dev, non_blocking=True) for k, v in y.items()}
            with torch.autocast(dev.type, enabled=amp):
                out = model(x)
            L = losses({k: v.float() for k, v in out.items()}, y, pos_weight)
            loss = sum(L.values())
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sched.step()
            for k, v in L.items():
                tot[k] = tot.get(k, 0) + v.item() / len(tl)
        res = evaluate(model, vl, dev)
        row = dict(epoch=ep + 1, sec=round(time.time() - t0, 1), **{f'loss_{k}': round(v, 4) for k, v in tot.items()},
                   **{k: round(v, 4) if isinstance(v, float) else v for k, v in res.items()})
        hist.append(row); print(json.dumps(row), flush=True)
        torch.save(dict(task=a.task, model=model.state_dict(), epoch=ep + 1, args=vars(a)), run_dir / 'last.pt')
        (run_dir / 'history.json').write_text(json.dumps(hist, indent=1))
    (run_dir / 'result_val.json').write_text(json.dumps(hist[-1], indent=1))
    print('done', run_dir / 'last.pt')


if __name__ == '__main__':
    main()
