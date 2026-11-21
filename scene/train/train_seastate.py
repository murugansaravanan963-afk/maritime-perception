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

CLASSES = ['calm', 'moderate', 'rough']
SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


class SeaStateData(Dataset):
    def __init__(self, data_dir, split, train, limit=0):
        self.data_dir = Path(data_dir)
        self.train = train
        labels = json.load(open(self.data_dir / 'seastate_labels.json'))
        self.records = [r for r in labels['records'] if r['split'] == split]
        if limit > 0:
            random.Random(0).shuffle(self.records)
            self.records = self.records[:limit]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        image = cv2.imread(str(self.data_dir / record['image']))
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.train:
            top = random.randint(0, image.shape[0] - SIZE)
            left = random.randint(0, image.shape[1] - SIZE)
        else:
            top = (image.shape[0] - SIZE) // 2
            left = (image.shape[1] - SIZE) // 2
        image = image[top:top + SIZE, left:left + SIZE]
        image = image.astype(np.float32) / 255.0
        if self.train:
            if random.random() < 0.5:
                image = image[:, ::-1]
            if random.random() < 0.3:
                image = image.mean(axis=2, keepdims=True).repeat(3, axis=2)
            mean = image.mean()
            image = (image - mean) * random.uniform(0.7, 1.3) + mean * random.uniform(0.8, 1.2)
            image = np.clip(image, 0.0, 1.0)
        image = (image - MEAN) / STD
        image = torch.from_numpy(image.transpose(2, 0, 1).copy())
        label = CLASSES.index(record['label'])
        return image, label


def build_model():
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(512, len(CLASSES))
    return model


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval()
    confusion = np.zeros((len(CLASSES), len(CLASSES)), dtype=np.int64)
    for images, labels in loader:
        outputs = model(images.to(device))
        predictions = outputs.argmax(dim=1).cpu()
        for true, pred in zip(labels.tolist(), predictions.tolist()):
            confusion[true, pred] += 1
    per_class = confusion.diagonal() / confusion.sum(axis=1).clip(min=1)
    accuracy = confusion.trace() / confusion.sum()
    return accuracy, per_class, confusion


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data', default='scene/data/uniri_seastate')
    parser.add_argument('--out', default='runs_scene/seastate')
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--batch', type=int, default=64)
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
    model = build_model().to(device)

    if args.test:
        model.load_state_dict(torch.load(out_dir / 'last.pt', map_location=device))
        test_data = SeaStateData(args.data, 'test', train=False, limit=args.limit)
        test_loader = DataLoader(test_data, batch_size=args.batch, num_workers=args.workers)
        accuracy, per_class, confusion = evaluate(model, test_loader, device)
        print('TEST images', len(test_data), 'accuracy', round(float(accuracy), 3))
        for name, value in zip(CLASSES, per_class):
            print(f'  {name:9s} {value:.3f}')
        print('confusion (rows = true, columns = predicted)')
        print(confusion)
        result = {'accuracy': float(accuracy), 'per_class': dict(zip(CLASSES, per_class.tolist())),
                  'confusion': confusion.tolist()}
        json.dump(result, open(out_dir / 'test_result.json', 'w'), indent=1)
        return

    train_data = SeaStateData(args.data, 'train', train=True, limit=args.limit)
    val_data = SeaStateData(args.data, 'val', train=False, limit=args.limit)
    print('train images', len(train_data), 'val images', len(val_data), 'device', device)
    train_loader = DataLoader(train_data, batch_size=args.batch, shuffle=True, num_workers=args.workers)
    val_loader = DataLoader(val_data, batch_size=args.batch, num_workers=args.workers)

    loss_function = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    history = []
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            loss = loss_function(outputs, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        mean_loss = total_loss / len(train_loader)
        accuracy, per_class, _ = evaluate(model, val_loader, device)
        print(f'epoch {epoch:3d}  loss {mean_loss:.3f}  val accuracy {accuracy:.3f}  per class {np.round(per_class, 3)}')
        history.append({'epoch': epoch, 'loss': mean_loss, 'val_accuracy': float(accuracy),
                        'val_per_class': per_class.tolist()})
        torch.save(model.state_dict(), out_dir / 'last.pt')
        json.dump(history, open(out_dir / 'history.json', 'w'), indent=1)
    print('saved', out_dir / 'last.pt')


if __name__ == '__main__':
    main()
