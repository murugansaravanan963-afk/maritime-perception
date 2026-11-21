"""Sea state v2: calm / moderate / rough, with a training recipe that fights overfitting.

What changed from train_seastate.py (v1):
  1. New validation split. The old val had 22 videos, almost no Beaufort 3 and no Beaufort 7-8,
     so choosing epochs on it was unreliable. Train+val videos are pooled and about 20% of the
     videos of each Beaufort level go to val (levels with fewer than 3 videos stay in train).
     The test split (UNIRI's own test videos) is never touched.
  2. The model predicts the Beaufort level (1-8) with "soft" targets: the true level gets most of
     the weight and its neighbours a little, because Beaufort 4 vs 5 is a small difference while
     1 vs 8 is a big one. calm / moderate / rough = sum of the Beaufort probabilities in each group.
  3. Each video gets the same weight when sampling, so the long videos are not memorised.
  4. Fine-tuning in two stages: first only the new head (backbone frozen), then everything with a
     10x smaller learning rate for the backbone. Plus dropout, weight decay and stronger colour aug.
  5. The best epoch (val balanced accuracy) is kept in best.pt, not just the last one.

  python scene/train/train_seastate_v2.py --out runs_scene/seastate_v2
  python scene/train/train_seastate_v2.py --out runs_scene/seastate_v2 --test
"""
import argparse
import collections
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler

CLASSES = ['calm', 'moderate', 'rough']
GROUP = {1: 0, 2: 0, 3: 1, 4: 1, 5: 2, 6: 2, 7: 2, 8: 2}     # Beaufort level -> class index
LEVELS = 8
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def resplit(records, val_fraction=0.2, seed=0):
    """Pool train+val videos and give about val_fraction of each Beaufort level's videos to val."""
    videos = collections.defaultdict(set)
    for r in records:
        if r['split'] in ('train', 'val'):
            videos[r['beaufort']].add(r['video'])
    val_videos = set()
    rng = random.Random(seed)
    for level, names in sorted(videos.items()):
        names = sorted(names)
        if len(names) < 3:
            continue
        rng.shuffle(names)
        val_videos.update(names[:max(1, round(len(names) * val_fraction))])
    for r in records:
        if r['split'] in ('train', 'val'):
            r['split'] = 'val' if r['video'] in val_videos else 'train'
    return records


class SeaStateData(Dataset):
    def __init__(self, data_dir, split, train, records):
        self.data_dir = Path(data_dir)
        self.train = train
        self.records = [r for r in records if r['split'] == split]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(self.data_dir / record['image']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.train:
            # random crop of 75-100% of the patch, resized to 224 (keeps the wave scale close to test)
            side = random.randint(int(image.shape[0] * 0.75), image.shape[0])
            top = random.randint(0, image.shape[0] - side)
            left = random.randint(0, image.shape[1] - side)
            image = cv2.resize(image[top:top + side, left:left + side], (SIZE, SIZE), interpolation=cv2.INTER_AREA)
        else:
            top = (image.shape[0] - SIZE) // 2
            left = (image.shape[1] - SIZE) // 2
            image = image[top:top + SIZE, left:left + SIZE]
        image = image.astype(np.float32) / 255.0
        if self.train:
            if random.random() < 0.5:
                image = image[:, ::-1]
            if random.random() < 0.2:
                image = image.mean(axis=2, keepdims=True).repeat(3, axis=2)
            if random.random() < 0.2:
                image = cv2.GaussianBlur(image, (5, 5), 0)
            image = image * np.random.uniform(0.85, 1.15, size=3).astype(np.float32)   # colour cast
            mean = image.mean()
            image = (image - mean) * random.uniform(0.7, 1.3) + mean * random.uniform(0.75, 1.25)
            image = np.clip(image, 0.0, 1.0)
        image = (image - MEAN) / STD
        image = torch.from_numpy(np.ascontiguousarray(image.transpose(2, 0, 1)))
        return image, record['beaufort'] - 1


def build_model(dropout=0.3):
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(512, LEVELS))
    return model


def soft_targets(levels, sigma=0.8):
    """Beaufort index (0-7) -> soft label: most weight on the true level, a little on its neighbours."""
    grid = torch.arange(LEVELS, device=levels.device).float()
    weights = torch.exp(-(grid[None, :] - levels[:, None].float()) ** 2 / (2 * sigma ** 2))
    return weights / weights.sum(dim=1, keepdim=True)


def class_probs(logits):
    """Beaufort logits (N, 8) -> calm / moderate / rough probabilities (N, 3)."""
    p = logits.softmax(dim=1)
    return torch.stack([p[:, 0:2].sum(1), p[:, 2:4].sum(1), p[:, 4:8].sum(1)], dim=1)


def video_balanced_sampler(records):
    """Every video has the same total weight inside its class, and every class the same total weight."""
    frames = collections.Counter(r['video'] for r in records)
    videos_per_class = collections.Counter()
    for video in frames:
        label = next(GROUP[r['beaufort']] for r in records if r['video'] == video)
        videos_per_class[label] += 1
    weights = [1.0 / frames[r['video']] / videos_per_class[GROUP[r['beaufort']]] for r in records]
    return WeightedRandomSampler(weights, num_samples=len(records), replacement=True)


def tta_views(images):
    """Test-time augmentation: the centre crop and its mirror image."""
    return [images, torch.flip(images, dims=[3])]


@torch.no_grad()
def evaluate(model, loader, device, tta=False):
    model.eval()
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    level_errors = []
    for images, levels in loader:
        images = images.to(device)
        views = tta_views(images) if tta else [images]
        logits = torch.stack([model(v) for v in views]).mean(0)
        predictions = class_probs(logits).argmax(dim=1).cpu()
        expected_level = (logits.softmax(1).cpu() * torch.arange(1, LEVELS + 1)).sum(1)
        level_errors += (expected_level - (levels + 1)).abs().tolist()
        for level, pred in zip(levels.tolist(), predictions.tolist()):
            confusion[GROUP[level + 1], pred] += 1
    per_class = confusion.diagonal() / confusion.sum(axis=1).clip(min=1)
    accuracy = confusion.trace() / confusion.sum()
    return accuracy, per_class.mean(), per_class, confusion, float(np.mean(level_errors))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='scene/data/uniri_seastate')
    parser.add_argument('--out', default='runs_scene/seastate_v2')
    parser.add_argument('--head-epochs', type=int, default=2)
    parser.add_argument('--epochs', type=int, default=25)
    parser.add_argument('--batch', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3, help='head learning rate; the backbone uses lr/10')
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    records = resplit(json.load(open(Path(args.data) / 'seastate_labels.json'))['records'])
    model = build_model().to(device)

    if args.test:
        model.load_state_dict(torch.load(out_dir / 'best.pt', map_location=device))
        test_data = SeaStateData(args.data, 'test', False, records)
        test_loader = DataLoader(test_data, batch_size=args.batch, num_workers=args.workers)
        result = {}
        for tta in (False, True):
            accuracy, balanced, per_class, confusion, level_mae = evaluate(model, test_loader, device, tta)
            name = 'flip_tta' if tta else 'single'
            print(f'TEST ({name}) images {len(test_data)}  accuracy {accuracy:.3f}  balanced {balanced:.3f}  '
                  f'Beaufort error {level_mae:.2f}')
            for label, value in zip(CLASSES, per_class):
                print(f'  {label:9s} {value:.3f}')
            print('confusion (rows = true, columns = predicted)')
            print(confusion)
            result[name] = {'accuracy': float(accuracy), 'balanced_accuracy': float(balanced),
                            'per_class': dict(zip(CLASSES, per_class.tolist())), 'confusion': confusion.tolist(),
                            'beaufort_mean_abs_error': level_mae}
        json.dump(result, open(out_dir / 'test_result.json', 'w'), indent=1)
        return

    train_data = SeaStateData(args.data, 'train', True, records)
    val_data = SeaStateData(args.data, 'val', False, records)
    split_info = {s: dict(collections.Counter(r['beaufort'] for r in records if r['split'] == s))
                  for s in ('train', 'val')}
    print('train images', len(train_data), 'val images', len(val_data), 'device', device)
    print('Beaufort levels per split', json.dumps(split_info, sort_keys=True))
    sampler = video_balanced_sampler(train_data.records)
    train_loader = DataLoader(train_data, batch_size=args.batch, sampler=sampler, num_workers=args.workers,
                              persistent_workers=args.workers > 0)
    val_loader = DataLoader(val_data, batch_size=args.batch, num_workers=args.workers,
                            persistent_workers=args.workers > 0)

    head = list(model.fc.parameters())
    backbone = [p for n, p in model.named_parameters() if not n.startswith('fc.')]
    optimizer = torch.optim.AdamW([{'params': backbone, 'lr': args.lr / 10}, {'params': head, 'lr': args.lr}],
                                  weight_decay=0.05)
    total = args.head_epochs + args.epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=[args.lr / 10, args.lr],
                                                    total_steps=total * len(train_loader), pct_start=0.1)
    scaler = torch.amp.GradScaler(enabled=device.type == 'cuda')

    history = []
    best = -1.0
    for epoch in range(1, total + 1):
        frozen = epoch <= args.head_epochs          # stage 1: only the new head learns
        model.train()
        for p in backbone:
            p.requires_grad = not frozen
        total_loss = 0.0
        for images, levels in train_loader:
            images = images.to(device, non_blocking=True)
            levels = levels.to(device, non_blocking=True)
            with torch.autocast(device.type, enabled=device.type == 'cuda'):
                logits = model(images)
            loss = -(soft_targets(levels) * F.log_softmax(logits.float(), dim=1)).sum(1).mean()
            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            total_loss += loss.item()
        mean_loss = total_loss / len(train_loader)
        accuracy, balanced, per_class, _, level_mae = evaluate(model, val_loader, device)
        mark = ''
        if balanced > best:
            best = balanced
            torch.save(model.state_dict(), out_dir / 'best.pt')
            mark = '  <- best'
        stage = 'head' if frozen else 'all'
        print(f'epoch {epoch:3d} ({stage})  loss {mean_loss:.3f}  val acc {accuracy:.3f}  balanced {balanced:.3f}  '
              f'per class {np.round(per_class, 3)}  Beaufort err {level_mae:.2f}{mark}', flush=True)
        history.append({'epoch': epoch, 'stage': stage, 'loss': mean_loss, 'val_accuracy': float(accuracy),
                        'val_balanced': float(balanced), 'val_per_class': per_class.tolist(),
                        'val_beaufort_error': level_mae})
        torch.save(model.state_dict(), out_dir / 'last.pt')
        json.dump({'split': split_info, 'history': history}, open(out_dir / 'history.json', 'w'), indent=1)
    print(f'best val balanced accuracy {best:.3f}  saved', out_dir / 'best.pt')


if __name__ == '__main__':
    main()
