# Part 4: The water-segmentation model

You already have a working `train_seastate.py`. The water model is **the same script with five changes**.
That is the main lesson of this part: once you have one training script, other models are small edits.

Start by copying your file:

```bash
copy scene\learn\train_seastate.py scene\learn\train_water.py
```

(In Git Bash: `cp scene/learn/train_seastate.py scene/learn/train_water.py`.)

## What changes and why

| | Sea state | Water segmentation |
|---|---|---|
| Question | "what sea state is this patch?" | "what is **each pixel**: obstacle, water or sky?" |
| Input | 224 x 224 water patch | whole photo, 640 x 384 |
| Answer (label) | one number (0, 1, 2) | a mask: one number **per pixel** (0, 1, 2, or 255 = ignore) |
| Model output | 3 scores | 3 scores **for every pixel**: shape (3, 384, 640) |
| Loss | cross-entropy | cross-entropy **per pixel**, skipping 255 |
| Score | accuracy | **IoU** (overlap between predicted and true region) |

Data: `scene/data/scene_lars_v1`: 2,312 train, 198 val, 293 test photos with masks (see `scene/README.md`).

---

## Lesson 4.1: Top of the file and the data class

Change the constants: the class list is now the 3 mask classes, and we add the ignore value.

```python
import torch.nn.functional as F
```
(add this under `import torch.nn as nn`)

```python
CLASSES = ['obstacle', 'water', 'sky']
IGNORE = 255
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
```

- `SIZE` is gone: the photos are already 640 x 384 (we letterboxed them when building the dataset).
- `import torch.nn.functional as F`: PyTorch's toolbox of operations that have no learnable weights,
  like resizing a feature map. Nickname `F` by convention.
- *Why 640 x 384?* Most of our photos are 16:9 (1920 x 1080 etc.). 640 x 360 keeps that shape;
  384 = 360 rounded up to a multiple of 32 (the model shrinks the image 5 times by 2, so the size must
  divide by 2^5 = 32). The 24 extra rows are grey padding marked "ignore".

Now the data class. Rename it `WaterData` and change three things:

```python
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
```

**The changed lines**

- `'scene_labels.json'`: the water dataset's labels file (lesson 1.6 showed one of its masks).
- `and r['mask'] is not None`: keep only photos that have a mask. `is not None` is how Python asks
  "is there something here?". (The LaRS-test photos have no public masks, so they are skipped.)
- `mask = cv2.imread(..., cv2.IMREAD_UNCHANGED)`: read the mask exactly as stored (lesson 1.6).
- **No crop**: we need the whole scene; the water edge can be anywhere.
- `mask = mask[:, ::-1]` **inside the same `if`** as the image flip.
  - *Why this matters:* if the image is mirrored but the mask is not, the "answer sheet" no longer matches
    the picture: the model is told the left shore is water. **Any change of position must be applied to the
    image and the label together.** Colour changes (brightness) only touch the image, because the class of a
    pixel doesn't depend on brightness.
- `image * random.uniform(0.8, 1.2)`: a simple brightness change, 80% to 120%.
- `mask.astype(np.int64)`: class numbers must be **whole numbers of type int64** for the loss function
  (it uses them as positions in a list of 3 scores; positions must be integers).
- Return `image, mask`: the "label" is now a whole grid.

**Check:** temporarily put this at the bottom of the file (like in Part 2) and run:

```python
if __name__ == '__main__':
    data = WaterData('scene/data/scene_lars_v1', 'train', train=True)
    image, mask = data[0]
    print(len(data), image.shape, mask.shape, mask.dtype, torch.unique(mask))
```

Expected (the last part can vary because of the random flip, not the values):

```
2312 torch.Size([3, 384, 640]) torch.Size([384, 640]) torch.int64 tensor([  0,   1,   2, 255])
```

---

## Lesson 4.2: The model, from "one answer" to "an answer per pixel"

ResNet18 shrinks the image step by step while it understands more and more. At the end it has a
12 x 20 grid of features. The sea-state model averaged that grid into one answer. Here we need the grid
back at full size. Replace `build_model` with this class:

```python
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
```

**First, two ideas you need**

- **Convolution (`Conv2d`)**: a small window (for example 3 x 3 pixels) slides over the image. At each
  position it multiplies the pixels by learned weights and adds them up into one number. One window learns
  to detect one pattern (an edge, a wave crest, a tree texture). A layer has many windows, so it produces
  many **feature maps** (one per pattern). The number of maps is the **channels**.
  - *Concept (DSA / signal processing):* this is a 2-D filter, like a blur or edge filter in image processing,
    except the filter values are learned instead of designed by hand.
- **Shrinking**: each ResNet stage halves the height and width and adds channels. This is what printing
  the shapes for one photo gives:

| After | Shape (channels, height, width) | Each cell sees |
|---|---|---|
| input | 3 x 384 x 640 | 1 pixel |
| `start` (incl. layer1) | 64 x 96 x 160 | small area: edges |
| `layer2` | 128 x 48 x 80 | textures (ripples, leaves) |
| `layer3` | 256 x 24 x 40 | parts (shoreline, hull) |
| `layer4` | 512 x 12 x 20 | big picture: "this is sky over a lake" |

Deep maps **know a lot but are blurry** (12 x 20 cells for the whole photo). Shallow maps are **sharp but
know little**. A good segmentation needs both.

**Line by line**

- `class WaterNet(nn.Module):` our own model class. `nn.Module` is PyTorch's base class for models;
  inheriting from it gives us saving, `.to(device)`, `.train()`/`.eval()` for free.
- `super().__init__()`: run `nn.Module`'s own setup first. Always the first line in a model's `__init__`.
- `resnet = torchvision.models.resnet18(weights='IMAGENET1K_V1')`: the same pretrained network as Part 3.
  We take its pieces apart instead of using it whole.
- `nn.Sequential(a, b, c, ...)`: glue layers into a chain: the output of one is the input of the next.
  `self.start` = everything up to and including `layer1`.
- `self.side2/3/4 = nn.Conv2d(C, 64, kernel_size=1)`: **1 x 1 convolutions** that turn 128, 256 or 512
  channels into 64, so the three levels can be **added** together (you can only add maps with the same shape).
- `self.head`: turns the combined 64 channels into **3 scores per cell** (obstacle, water, sky).
  - `Conv2d(64, 64, kernel_size=3, padding=1)`: a 3 x 3 window to smooth and mix neighbours.
    `padding=1` adds a 1-pixel border so the size stays the same.
  - `BatchNorm2d`: keeps the numbers in a stable range during training (training is faster and steadier).
  - `ReLU`: keep positive numbers, turn negatives into 0. Without such a "non-linear" step, many layers in a
    row would collapse into one big multiplication and the network couldn't learn complicated shapes.
  - The last `Conv2d(64, 3, kernel_size=1)`: 3 output channels = 3 classes.
- `def forward(self, x):` what happens when data goes through: PyTorch calls it when you write `model(x)`.
- `f2, f3, f4`: keep the feature maps of three stages.
- `p = self.side4(f4)`: start from the deepest, blurriest map (12 x 20).
- `F.interpolate(p, scale_factor=2)`: **upsample** to 24 x 40 (each cell copied into a 2 x 2 block), then
  `+ self.side3(f3)`: **add** the sharper map of the same size. This is a **skip connection**: detail from
  an early layer "skips" past the deep layers and is added back.
- Again: up to 48 x 80 and add `side2(f2)`.
- `self.head(p)`: 3 scores per cell on a 48 x 80 grid.
- `F.interpolate(p, scale_factor=8, mode='bilinear', align_corners=False)`: stretch to 384 x 640 so there is
  one set of 3 scores **per pixel**, the same size as the mask. `bilinear` blends smoothly between cells
  instead of making blocks.
  - *Why stop at 48 x 80 and stretch, instead of going all the way up with more layers?* Water / sky / obstacle
    regions are big and smooth; 8-pixel detail is enough, and fewer layers = faster on the Hailo.
- *Concept (ML):* this design is called **FCN** (fully convolutional network) with skip connections, the
  classic segmentation design. Fancier versions (U-Net, DeepLab) use the same idea.

The model has 11.3 million learnable numbers, almost all from the pretrained ResNet18.

---

## Lesson 4.3: Loss and score

**Loss:** one change in `main`:

```python
    loss_function = nn.CrossEntropyLoss(ignore_index=IGNORE)
```

- The **same cross-entropy** as for sea state, but computed for every pixel and averaged.
  The output is (batch, 3, 384, 640) and the mask (batch, 384, 640): PyTorch understands that each pixel
  is its own little 3-class question.
- `ignore_index=IGNORE`: pixels with value 255 are left out of the average: no reward or punishment there.

**Score:** replace `evaluate` with an **IoU** version:

```python
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
```

- **IoU (Intersection over Union)** for one class, e.g. water: take the pixels the model *called* water and the
  pixels that *are* water.
  - **Intersection** = pixels in both (correct water). **Union** = pixels in either.
  - IoU = intersection / union. 1.0 = perfect overlap; 0 = no overlap.
  - *Why not just "% of pixels correct"?* Sky is often half the photo. A model that only ever says "sky" could get
    50% pixel accuracy and still be useless. IoU scores **each class separately**, so a tiny class counts as much as a big one.
  - It's the same IoU you know from detection boxes, but for regions of any shape.
- `.argmax(dim=1)`: for each pixel, the class with the highest score: a predicted mask.
- `valid = masks != IGNORE`: True where we have an answer. `&` = "and" for whole grids, `|` = "or".
- `.sum().item()`: count the True pixels and turn the result into a plain Python number.
- `np.maximum(union, 1)`: avoid dividing by 0 if a class never appears.
- `iou.mean()` = **mIoU**, the mean over the 3 classes: our headline score.

**In `main`, change:**

- The defaults: `--data scene/data/scene_lars_v1`, `--out runs_scene/water`, `--epochs 40`, `--batch 16`.
  - *Why a smaller batch (16 vs 64)?* Each photo is 640 x 384 instead of 224 x 224 (about 5 times the
    pixels) and the output is a full map. 16 fits comfortably in GPU memory.
  - *Why more epochs (40 vs 20)?* Fewer images (2,312) and a harder, per-pixel task: it needs more passes.
- `model = WaterNet().to(device)`; `WaterData` instead of `SeaStateData`.
- The loop variable `labels` becomes `masks` (same code, clearer name).
- The print line: `val mean IoU {miou:.3f}  per class {np.round(iou, 3)}`, and the test printout lists the
  IoU per class name.

Compare with `scene/learn/solutions/train_water.py` when you are done.

---

## Lesson 4.4: Run it on the laptop

```bash
python scene/learn/train_water.py --epochs 2 --batch 4 --workers 0 --limit 16
python scene/learn/train_water.py --test --batch 4 --workers 0 --limit 16
```

Expected (CPU, about 1-2 minutes; your numbers will differ a little):

```
train images 16 val images 16 device cpu
epoch   1  loss 0.770  val mean IoU 0.724  per class [0.526 0.838 0.809]
epoch   2  loss 0.372  val mean IoU 0.807  per class [0.651 0.877 0.892]
saved runs_scene\water\last.pt
TEST images 16 mean IoU 0.728
  obstacle  0.602
  water     0.739
  sky       0.844
```

Already 0.8 mIoU after 2 tiny epochs: the pretrained ResNet18 already "knows" what sky and water look like.
**Delete `runs_scene/water` before the real training on Sol** (this run used 16 images; don't mix it up).

**Your turn**

1. Remove `mask = mask[:, ::-1]` (keep the image flip), train 2 epochs with `--limit 16` and compare the val
   mIoU. Then put it back. What did you learn?
2. Remove `ignore_index=IGNORE`. Run. Read the error. Why does a value of 255 break a 3-class loss?
3. Print `p.shape` after each step in `forward` (temporarily) and check it against the table in lesson 4.2.

## Answers

1. Usually lower and less stable: about half of the training images now have mirrored answers. On 16 images
   and 2 epochs the difference is noisy; on the full data it is clear. Lesson: **position changes go to image and label together.**
2. An error saying the target 255 is out of bounds: the loss uses the label as a position in the list of 3
   scores (0, 1, 2); position 255 doesn't exist.
3. 64x12x20 → 64x24x40 → 64x48x80 → 3x48x80 → 3x384x640.
