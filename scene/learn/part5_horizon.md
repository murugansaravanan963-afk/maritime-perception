# Part 5: The horizon model

The horizon model predicts **two numbers**: the height of the water-edge line at the **left** edge of the
photo and at the **right** edge. Two points define a straight line, so that is the whole horizon.

Again: copy a script and change it. Start from your **water** script (the data side is closest):

```bash
copy scene\learn\train_water.py scene\learn\train_horizon.py
```

## What changes and why

| | Water segmentation | Horizon |
|---|---|---|
| Answer | a class per pixel | **2 numbers**: y_left, y_right |
| Kind of problem | classification (pick a class) | **regression** (predict a number) |
| Model output | (3, 384, 640) | (2,) |
| Loss | cross-entropy | **L1** (average distance) |
| Score | IoU | **error in pixels**, % within 10 px |
| Photos | all with a mask | only where the water edge is really a line (`line_ok`) |

*Where do the answers come from?* Nobody drew horizon lines. `build_scene_data.py` takes each mask, finds the
top-most water pixel in every column, and fits a straight line through those points (ignoring columns where
a boat hides the edge). Narrow canals with V-shaped banks are not a line, so they are marked
`line_ok = False` and left out. 2,028 train, 152 val, 232 test photos remain.

---

## Lesson 5.1: Constants and the data class

```python
HEIGHT = 384
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
```

(`CLASSES` and `IGNORE` are gone: there are no classes. You no longer need `import torch.nn.functional as F` either.)

```python
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
```

**The changed lines**

- The filter now has three conditions joined by `and`. Python checks them left to right and **stops at the
  first False**. That order matters: if `r['horizon']` is `None`, asking `r['horizon']['line_ok']` would
  crash, but Python never gets there. (This is called *short-circuit* evaluation.)
- The long line is split over two lines inside the `[...]`: Python allows a line break inside brackets.
- `y_left = record['horizon']['y_left'] / HEIGHT`: the line height in pixels (0 = top, 384 = bottom), divided
  by 384 to get **0.0-1.0**.
  - *Why divide?* The network's numbers live around -1...1. Asking it to output "193.5" directly means huge
    errors at the start and unstable training. Small targets are easier; we multiply by 384 again when we
    report the error in pixels.
- `y_left, y_right = y_right, y_left`: **swap** the two values in one line (Python evaluates the right side
  first, then assigns).
  - *Why swap when flipping?* After a mirror, the left edge of the photo becomes the right edge, so the
    line's left height is now its right height. **This is lesson 4.1 again: a position change must also change
    the answer.** Forget it and half the training images teach the wrong tilt.
- `torch.tensor([y_left, y_right], dtype=torch.float32)`: the answer is now a small tensor of two decimals.

**Check** (`train=False`, so no random flip):

```python
if __name__ == '__main__':
    data = HorizonData('scene/data/scene_lars_v1', 'train', train=False)
    image, target = data[0]
    print(len(data), image.shape, target, data.records[0]['id'])
```

Expected:

```
2028 torch.Size([3, 384, 640]) tensor([0.5038, 0.5524]) yt021_02_00049
```

0.5038 x 384 = 193.5 px on the left, 0.5524 x 384 = 212.1 px on the right: a line sloping slightly downwards
to the right, about the middle of the photo.

---

## Lesson 5.2: The model, why we must not average

In the sea-state model, ResNet18 ends with **average pooling**: the 12 x 20 grid of features is averaged
into one list of 512 numbers. That throws away **where** things are. For "is the sea rough?" location doesn't
matter. For "**where** is the horizon?" location is the whole question. So this model keeps the grid:

```python
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
```

**Line by line**

- `self.features`: all of ResNet18 **except** the average pooling and the final `fc` layer. Output: 512 x 12 x 20.
- `self.squeeze`: a 1 x 1 convolution from 512 to 32 channels.
  - *Why?* Next we connect every number to every neuron. 512 x 12 x 20 = 122,880 numbers would need 31 million
    weights in the next layer: slow and easy to overfit with 2,028 photos. 32 x 12 x 20 = 7,680 is plenty.
- `nn.Flatten()`: turns the 32 x 12 x 20 grid into one long row of 7,680 numbers. Each number still "remembers"
  its place, because **its position in the row is fixed**: number 0 is always the top-left cell of channel 0.
  So the next layer can learn "high values in row 6 of the grid = horizon near the middle".
- `nn.Linear(7680, 256)`: a **fully connected** layer: each of 256 outputs is a weighted sum of all 7,680 inputs.
- `nn.ReLU()`: the non-linear step (lesson 4.2).
- `nn.Linear(256, 2)`: two outputs: y_left and y_right. **No softmax**: these are plain numbers, not class scores.
- `forward`: features → squeeze → head, written as one nested line.

Output shape for a batch of 1 photo: `(1, 2)`. The model has 13.2 million learnable numbers (ResNet18's 11.2 M
plus 2 M in the new head).

*Concept (ML):* predicting numbers instead of classes is called **regression**. Your detector does both: class
scores (classification) and box coordinates (regression).

---

## Lesson 5.3: Loss and score

```python
    loss_function = nn.L1Loss()
```

- **L1 loss** = the average absolute difference |predicted - true| (the "distance").
  - *Why not cross-entropy?* There are no classes. Cross-entropy compares *probabilities over classes*;
    here we compare two numbers.
  - *Why L1 and not L2 (squared difference, `nn.MSELoss`)?* Squaring makes big errors count enormously.
    Some of our horizon answers are slightly wrong (they were fitted automatically). With L2 a few bad labels
    would dominate training; L1 treats them more calmly. L1 is "robust to outliers".

```python
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
```

- `(outputs - targets).abs()`: distance for both points; `.mean(dim=1)`: average of left and right for each photo;
  `* HEIGHT`: back to pixels.
- `errors.extend(...)`: add all values of a list to the end of `errors` (`append` would add the list as one item).
- Returns the **mean error in pixels** (lower is better) and the **fraction of photos within 10 px**
  (higher is better). 10 px on a 384-pixel-tall image is 2.6% of the height.
- *Why two numbers?* The mean can be ruined by a few very wrong photos; "% within 10 px" tells you how often it is
  simply right.

**In `main`:** `--out runs_scene/horizon`, keep `--epochs 40` and `--batch 16`, use `HorizonNet` and `HorizonData`,
rename `masks` → `targets`, and print `val error {mean_error:.1f} px  within 10 px {within_10:.3f}`.
The loss is small now (distances on a 0-1 scale), so print it with `:.4f`.

---

## Lesson 5.4: Run it on the laptop

```bash
python scene/learn/train_horizon.py --epochs 2 --batch 4 --workers 0 --limit 16
python scene/learn/train_horizon.py --test --batch 4 --workers 0 --limit 16
```

Expected (about 1 minute):

```
train images 16 val images 16 device cpu
epoch   1  loss 0.7454  val error 101.0 px  within 10 px 0.000
epoch   2  loss 0.4024  val error 101.4 px  within 10 px 0.000
saved runs_scene\horizon\last.pt
TEST images 16 mean error 116.6 px within 10 px 0.000
```

**This is expected to be bad.** Unlike the water model, the new head (`squeeze` + two `Linear` layers) starts
from random numbers and 16 photos x 2 epochs teach it almost nothing. The loss going down (0.75 → 0.40) is the
only thing to check here: it means the pipeline works. The real result comes from Sol (40 epochs, 2,028 photos).
Delete `runs_scene/horizon` afterwards.

**Your turn**

1. Remove the line that swaps `y_left, y_right`. Why can't you see the damage in this 2-epoch run, and how would
   you prove it is needed? (Think: what should happen to a flipped image's line?)
2. Change `nn.L1Loss()` to `nn.MSELoss()` and run 2 epochs. The loss numbers get much smaller. Is the model better?
3. A horizon line goes from y = 100 (left) to y = 140 (right). After a mirror flip, what are the two values?

## Answers

1. 2 epochs on 16 photos is far too little to learn tilt, so both versions are bad. To prove it, draw the line on a
   flipped image with and without the swap (like `horizon_check` in the dataset build): without the swap the line
   tilts the wrong way. On the full training the swap-less model learns a "flatter", wrong tilt.
2. Not necessarily. MSE squares numbers below 1, so 0.4 becomes 0.16: the *loss* looks smaller because it's a
   different ruler. Only the **error in pixels** can compare the two.
3. Left 140, right 100.
