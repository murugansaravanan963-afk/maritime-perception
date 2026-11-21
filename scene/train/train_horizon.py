import argparse
import json
import random
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision
from torch.utils.data import Dataset, DataLoader

HEIGHT = 384
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class HorizonData(Dataset):
    def __init__(self, data_dir, split, train, limit=0):
        self.data_dir = Path(data_dir)
        self.train = train
        labels = json.load(open(self.data_dir / 'scene_labels.json'))
        self.records = [r for r in labels['records']
                        if r['split'] == split and r['horizon'] is not None and r['horizon']['line_ok']]
        if limit > 0:
            random.Random(0).shuffle(self.records)
            self.records = self.records[:limit]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(self.data_dir / record['image']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image = image.astype(np.float32) / 255.0
        y_left = record['horizon']['y_left'] / HEIGHT
        y_right = record['horizon']['y_right'] / HEIGHT
        if self.train:
            if random.random() < 0.5:
                image = image[:, ::-1]
                y_left, y_right = y_right, y_left
            image = np.clip(image * random.uniform(0.8, 1.2), 0.0, 1.0)
        image = (image - MEAN) / STD
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        target = torch.tensor([y_left, y_right], dtype=torch.float32)
        return image, target


class HorizonNet(nn.Module):
    def __init__(self):
        super().__init__()
        resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool,
                                      resnet.layer1, resnet.layer2, resnet.layer3, resnet.layer4)
        self.squeeze = nn.Sequential(nn.Conv2d(512, 32, kernel_size=1), nn.ReLU())
        self.head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 12 * 20, 256),
            nn.ReLU(),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        return self.head(self.squeeze(self.features(x)))


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    errors = []
    for images, targets in loader:
        outputs = model(images.to(device)).cpu()
        error_px = (outputs - targets).abs().mean(dim=1) * HEIGHT
        errors.extend(error_px.tolist())
    errors = np.array(errors)
    return errors.mean(), (errors <= 10).mean()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='scene/data/scene_lars_v1')
    parser.add_argument('--out', default='runs_scene/horizon')
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
    model = HorizonNet().to(device)

    if args.test:
        model.load_state_dict(torch.load(out_dir / 'last.pt', map_location=device))
        test_data = HorizonData(args.data, 'test', train=False, limit=args.limit)
        test_loader = DataLoader(test_data, batch_size=args.batch, num_workers=args.workers)
        mean_error, within_10 = evaluate(model, test_loader, device)
        print('TEST images', len(test_data), f'mean error {mean_error:.1f} px', f'within 10 px {within_10:.3f}')
        result = {'mean_error_px': float(mean_error), 'within_10px': float(within_10)}
        json.dump(result, open(out_dir / 'test_result.json', 'w'), indent=1)
        return

    train_data = HorizonData(args.data, 'train', train=True, limit=args.limit)
    val_data = HorizonData(args.data, 'val', train=False, limit=args.limit)
    print('train images', len(train_data), 'val images', len(val_data), 'device', device)
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_data, batch_size=args.batch, num_workers=args.workers)

    loss_function = nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, targets in train_loader:
            images = images.to(device)
            targets = targets.to(device)
            outputs = model(images)
            loss = loss_function(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        mean_loss = total_loss / len(train_loader)
        mean_error, within_10 = evaluate(model, val_loader, device)
        print(f'epoch {epoch:3d}  loss {mean_loss:.4f}  val error {mean_error:.1f} px  within 10 px {within_10:.3f}')
        history.append({'epoch': epoch, 'loss': mean_loss, 'val_error_px': float(mean_error),
                        'val_within_10px': float(within_10)})
        torch.save(model.state_dict(), out_dir / 'last.pt')
        json.dump(history, open(out_dir / 'history.json', 'w'), indent=1)
    print('saved', out_dir / 'last.pt')


if __name__ == '__main__':
    main()
