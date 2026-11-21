import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.data import Dataset, DataLoader

CLASSES = ['obstacle', 'water', 'sky']
IGNORE = 255
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class WaterData(Dataset):
    def __init__(self, data_dir, split, train, limit=0):
        self.data_dir = Path(data_dir)
        self.train = train
        labels = json.load(open(self.data_dir / 'scene_labels.json'))
        self.records = [r for r in labels['records'] if r['split'] == split and r['mask'] is not None]
        if limit > 0:
            random.Random(0).shuffle(self.records)
            self.records = self.records[:limit]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(self.data_dir / record['image']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mask = cv2.imread(str(self.data_dir / record['mask']), cv2.IMREAD_UNCHANGED)
        image = image.astype(np.float32) / 255.0
        if self.train:
            if random.random() < 0.5:
                image = image[:, ::-1]
                mask = mask[:, ::-1]
            image = np.clip(image * random.uniform(0.8, 1.2), 0.0, 1.0)
        image = (image - MEAN) / STD
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        mask = torch.from_numpy(mask.astype(np.int64).copy())
        return image, mask


class WaterNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        self.start = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool, resnet.layer1)
        self.layer2 = resnet.layer2
        self.layer3 = resnet.layer3
        self.layer4 = resnet.layer4
        self.side2 = nn.Conv2d(128, 64, kernel_size=1)
        self.side3 = nn.Conv2d(256, 64, kernel_size=1)
        self.side4 = nn.Conv2d(512, 64, kernel_size=1)
        self.head = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, len(CLASSES), kernel_size=1),
        )

    def forward(self, x):
        x = self.start(x)
        f2 = self.layer2(x)
        f3 = self.layer3(f2)
        f4 = self.layer4(f3)
        p = self.side4(f4)
        p = F.interpolate(p, scale_factor=2) + self.side3(f3)
        p = F.interpolate(p, scale_factor=2) + self.side2(f2)
        p = self.head(p)
        return F.interpolate(p, scale_factor=8, mode='bilinear', align_corners=False)


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    intersection = np.zeros(len(CLASSES))
    union = np.zeros(len(CLASSES))
    for images, masks in loader:
        predictions = model(images.to(device)).argmax(dim=1).cpu()
        valid = masks != IGNORE
        for c in range(len(CLASSES)):
            pred_c = (predictions == c) & valid
            true_c = (masks == c) & valid
            intersection[c] += (pred_c & true_c).sum().item()
            union[c] += (pred_c | true_c).sum().item()
    iou = intersection / np.maximum(union, 1)
    return iou.mean(), iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='scene/data/scene_lars_v1')
    parser.add_argument('--out', default='runs_scene/water')
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--lr', type=float, default=3e-4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--limit', type=int, default=0)
    parser.add_argument('--test', action='store_true')
    args = parser.parse_args()

    random.seed(0)
    torch.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = WaterNet().to(device)

    if args.test:
        model.load_state_dict(torch.load(out_dir / 'last.pt', map_location=device))
        test_data = WaterData(args.data, 'test', train=False, limit=args.limit)
        test_loader = DataLoader(test_data, batch_size=args.batch, num_workers=args.workers)
        miou, iou = evaluate(model, test_loader, device)
        print('TEST images', len(test_data), 'mean IoU', round(float(miou), 3))
        for name, value in zip(CLASSES, iou):
            print(f'  {name:9s} {value:.3f}')
        result = {'mean_iou': float(miou), 'iou': dict(zip(CLASSES, iou.tolist()))}
        json.dump(result, open(out_dir / 'test_result.json', 'w'), indent=1)
        return

    train_data = WaterData(args.data, 'train', train=True, limit=args.limit)
    val_data = WaterData(args.data, 'val', train=False, limit=args.limit)
    print('train images', len(train_data), 'val images', len(val_data), 'device', device)
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_data, batch_size=args.batch, num_workers=args.workers)

    loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, masks in train_loader:
            images = images.to(device)
            masks = masks.to(device)
            outputs = model(images)
            loss = loss_function(outputs, masks)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        mean_loss = total_loss / len(train_loader)
        miou, iou = evaluate(model, val_loader, device)
        print(f'epoch {epoch:3d}  loss {mean_loss:.3f}  val mean IoU {miou:.3f}  per class {np.round(iou, 3)}')
        history.append({'epoch': epoch, 'loss': mean_loss, 'val_mean_iou': float(miou), 'val_iou': iou.tolist()})
        torch.save(model.state_dict(), out_dir / 'last.pt')
        json.dump(history, open(out_dir / 'history.json', 'w'), indent=1)
    print('saved', out_dir / 'last.pt')


if __name__ == '__main__':
    main()
