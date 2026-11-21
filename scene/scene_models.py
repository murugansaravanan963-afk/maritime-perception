"""Scene models: water segmentation, horizon, sea state, and the combined multi-task model.

All four share one ResNet18 backbone design (ImageNet-pretrained, torchvision), input 384x640
RGB, ImageNet mean/std normalized. Only Conv / BN / ReLU / resize / add / pooling / dense layers,
so they compile for the Hailo-8L.

  seg      -> logits (3, 96, 160): obstacle, water, sky at 1/4 resolution
  horizon  -> (2,): y of the water-edge line at x=0 and x=639, divided by 384
  state    -> (4,): logits for calm, fog, glitter, low_light
  multi    -> all three from one shared backbone (baseline B3)
"""
import torch, torch.nn as nn, torch.nn.functional as F
import torchvision

H, W = 384, 640
NUM_SEG, NUM_STATE = 3, 4


def cbr(i, o, k=3):
    return nn.Sequential(nn.Conv2d(i, o, k, padding=k // 2, bias=False), nn.BatchNorm2d(o), nn.ReLU(inplace=True))


class Backbone(nn.Module):
    def __init__(self, pretrained=True):
        super().__init__()
        r = torchvision.models.resnet18(weights='IMAGENET1K_V1' if pretrained else None)
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu, r.maxpool)
        self.l1, self.l2, self.l3, self.l4 = r.layer1, r.layer2, r.layer3, r.layer4

    def forward(self, x):
        c2 = self.l1(self.stem(x))   # 64  x 96 x 160
        c3 = self.l2(c2)             # 128 x 48 x 80
        c4 = self.l3(c3)             # 256 x 24 x 40
        c5 = self.l4(c4)             # 512 x 12 x 20
        return c2, c3, c4, c5


class SegHead(nn.Module):
    """FPN-lite decoder, output at 1/4 resolution."""
    def __init__(self, ch=96):
        super().__init__()
        self.lat = nn.ModuleList([nn.Conv2d(c, ch, 1) for c in (64, 128, 256, 512)])
        self.smooth = nn.ModuleList([cbr(ch, ch) for _ in range(3)])
        self.out = nn.Sequential(cbr(ch, ch), nn.Conv2d(ch, NUM_SEG, 1))

    def forward(self, feats):
        p = self.lat[3](feats[3])
        for i in (2, 1, 0):
            p = F.interpolate(p, scale_factor=2, mode='nearest') + self.lat[i](feats[i])
            p = self.smooth[i](p)
        return self.out(p)


class HorizonHead(nn.Module):
    """Keeps spatial layout (no global pooling), so the line position is learnable."""
    def __init__(self):
        super().__init__()
        self.reduce = cbr(512, 32, 1)
        self.fc = nn.Sequential(nn.Flatten(), nn.Linear(32 * (H // 32) * (W // 32), 256),
                                nn.ReLU(inplace=True), nn.Linear(256, 2))

    def forward(self, feats):
        return self.fc(self.reduce(feats[3]))


class StateHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc = nn.Sequential(nn.AdaptiveAvgPool2d(1), nn.Flatten(), nn.Dropout(0.2), nn.Linear(512, NUM_STATE))

    def forward(self, feats):
        return self.fc(feats[3])


HEADS = {'seg': SegHead, 'horizon': HorizonHead, 'state': StateHead}
TASK_HEADS = {'seg': ['seg'], 'horizon': ['horizon'], 'state': ['state'], 'multi': ['seg', 'horizon', 'state']}


class SceneModel(nn.Module):
    def __init__(self, task, pretrained=True):
        super().__init__()
        self.task = task
        self.backbone = Backbone(pretrained)
        self.heads = nn.ModuleDict({h: HEADS[h]() for h in TASK_HEADS[task]})

    def forward(self, x):
        f = self.backbone(x)
        return {h: m(f) for h, m in self.heads.items()}


class ExportWrapper(nn.Module):
    """ONNX export: tuple outputs in a fixed order (seg, horizon, state; whichever exist)."""
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, x):
        out = self.model(x)
        return tuple(out[h] for h in ('seg', 'horizon', 'state') if h in out)
