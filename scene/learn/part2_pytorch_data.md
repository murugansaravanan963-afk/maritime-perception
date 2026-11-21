# Part 2: Feeding data to PyTorch

From here on you build **one real file**: `scene/learn/train_seastate.py`. Part 2 writes its first half
(the data). Part 3 adds the model and the training. Install PyTorch first (see `START_HERE.md`).

---

## Lesson 2.1: Tensors

**Goal:** meet PyTorch's version of a NumPy array. (Scratch file `scene/learn/lesson21.py`.)

```python
import numpy as np
import torch

a = np.zeros((224, 224, 3), dtype=np.float32)
t = torch.from_numpy(a.transpose(2, 0, 1).copy())
print(t.shape, t.dtype)

batch = torch.stack([t, t, t, t])
print(batch.shape)

labels = torch.tensor([0, 2, 2, 1])
print(labels, labels.shape, labels.dtype)
print(torch.cuda.is_available())
```

**Line by line**

- `import torch` loads **PyTorch**, the library we build and train models with.
  - *Why PyTorch and not TensorFlow?* Both work. PyTorch is what Ultralytics (your detector) is built on, what most
    research code uses, and it exports to ONNX for the Hailo compiler.
- A **tensor** is PyTorch's array: same idea as a NumPy array (a grid of numbers with a shape), plus two extra
  powers: it can live **on the GPU**, and it can **remember the calculations done to it** so PyTorch can work out
  how to improve the model (Part 3).
- `torch.from_numpy(...)` turns a NumPy array into a tensor (sharing the same memory, no copy).
- `.copy()` after `transpose`: `transpose` only changes how NumPy *reads* the memory, not the memory itself.
  `from_numpy` needs the numbers laid out in the new order, so `.copy()` writes them out properly.
  Without it you may get an error about "negative strides" after a flip. Rule: **always `.copy()` before `from_numpy`
  if you transposed or flipped.**
- `torch.stack([t, t, t, t])` puts 4 images on top of each other: a **batch** with shape
  `(4, 3, 224, 224)` = (how many images, colours, height, width). Models always take batches.
  - *Why batches?* A GPU is like a factory with thousands of workers: giving it one image at a time leaves most
    of them idle. 64 images at once is almost as fast as 1.
- `torch.tensor([0, 2, 2, 1])`: the labels of 4 images as a tensor. `int64` = whole numbers, needed for class labels.
- `torch.cuda.is_available()`: is there an NVIDIA GPU? `False` on your laptop, `True` on Sol.

**Expected output**

```
torch.Size([3, 224, 224]) torch.float32
torch.Size([4, 3, 224, 224])
tensor([0, 2, 2, 1]) torch.Size([4]) torch.int64
False
```

---

## Lesson 2.2: The Dataset, start of the real script

**Goal:** a class that, given a number *i*, returns training image *i* as a ready tensor **and** its label.
This is your `SeaStateList` from lesson 0.8 grown up: same three methods, plus everything from Part 1.

Create `scene/learn/train_seastate.py`:

```python
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


if __name__ == '__main__':
    random.seed(0)
    data = SeaStateData('scene/data/uniri_seastate', 'train', train=True)
    print(len(data))
    image, label = data[0]
    print(image.shape, image.dtype)
    print(label, CLASSES[label])
    print(round(float(image.mean()), 3), round(float(image.std()), 3))
```

**The imports (top of the file)**

- All imports go at the top so a reader sees at once what the file depends on. Python's own libraries first
  (`argparse`, `json`, `random`, `pathlib`), then a blank line, then installed libraries. `argparse` is used in
  Part 3; importing it now is fine.
- `import torch.nn as nn`: PyTorch's building blocks for models (layers, loss functions).
- `import torchvision`: PyTorch's companion for images; it holds the pretrained ResNet18.
- `from torch.utils.data import Dataset, DataLoader`: the two data tools of this part.

**The constants**

- `CLASSES`, `SIZE`, `MEAN`, `STD`: lessons 0.3, 1.3 and 1.4. Defined once at the top so every function uses the
  same values. If `SIZE` were typed as 224 in three places and you changed one, you'd get a mismatch bug.

**`class SeaStateData(Dataset):`**

- `(Dataset)` means our class **inherits** from PyTorch's `Dataset`: we promise to provide `__len__` and
  `__getitem__`, and in return PyTorch's `DataLoader` knows how to use our class. A contract.
  - *Concept (firmware):* like implementing a driver interface: the OS doesn't care how your driver reads the
    hardware, only that it provides `read()` and `size()`.

**`__init__` (runs once)**

- `split` is `'train'`, `'val'` or `'test'`; `train` (True/False) says whether to augment. They're separate because
  you might want, say, the training images *without* augmentation to check something.
- `self.train = train`: store it on the object so `__getitem__` can see it later.
- `self.records = [...]`: lesson 0.7: keep only this split's records. **Only the labels are loaded here, not
  the images.** 8,014 images would be ~2.5 GB in memory as numbers; we read each one only when it's needed.
- `limit=0` is a **default value**: if you don't pass `limit`, it is 0 = "use everything". With `--limit 16`
  (Part 3) we test the whole script on 16 images in seconds.
- `random.Random(0).shuffle(self.records)`: mix the order with a **fixed seed** (0) before cutting, so the 16
  images are a mix of classes (the file is sorted by class) and the same 16 every time.
  - *Concept:* computers make "random" numbers with a formula; the **seed** is its starting value. Same seed =
    same sequence. That makes experiments repeatable.

**`__getitem__` (runs for every image, every epoch)**

- Read and convert to RGB: lesson 1.2.
- **Crop** (lesson 1.3): during training a **random** 224 x 224 window; for val/test always the **centre**.
  - *Why random for training?* Each epoch the model sees a slightly different part of the same sea: it can't
    memorise exact pixels and must learn the waves. Free extra variety.
  - *Why the centre for val/test?* Scores must be the same every time you measure. Random test crops would give a
    different accuracy on each run.
  - `random.randint(0, 331 - 224)` gives a whole number from 0 to 107 (both ends included).
- `/ 255.0`: lesson 1.4.
- Augmentation, **only when `self.train` is True** (lesson 1.5):
  - mirror with 50% chance (`random.random()` gives a decimal between 0 and 1; `< 0.5` is true half the time);
  - greyscale with 30% chance;
  - contrast ×0.7 to ×1.3 and brightness ×0.8 to ×1.2 (the `mean * random.uniform(...)` part shifts the
    average brightness), then `clip` back into 0-1.
  - *Why only for training?* Augmentation is practice with variations. The exam (val/test) must be the real
    images, unchanged.
- Normalise, channels first, `.copy()`, tensor: lessons 1.4 and 2.1.
- `label = CLASSES.index(record['label'])`: word → number (lesson 0.3). `'moderate'` → 1.
- `return image, label`: the pair the training needs.

**`if __name__ == '__main__':`**

- `__name__` is a special variable. When you **run** the file, it is `'__main__'`; when another file **imports**
  it, it is the file's name. So this block runs only when you run this file directly.
  - *Why we need it:* (1) it's where our quick checks go now (and `main()` in Part 3); (2) on Windows, the
    `DataLoader` with `num_workers > 0` starts helper processes that import this file. Without the guard, each
    helper would start training again. Endless loop.

**Run:** `python scene/learn/train_seastate.py`

**Expected output**

```
5036
torch.Size([3, 224, 224]) torch.float32
1 moderate
0.008 0.249
```

(`random.seed(0)` fixes the random crop and augmentation, so your numbers should match. Change the seed and the last
line changes, the rest doesn't.)

---

## Lesson 2.3: The DataLoader, batches for free

**Goal:** let PyTorch pick images, shuffle them, group them into batches and load them in parallel.

Add to the end of the `if __name__ == '__main__':` block:

```python
    loader = DataLoader(data, batch_size=8, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))
    print(images.shape)
    print(labels)
```

**Line by line**

- `DataLoader(data, batch_size=8, shuffle=True, num_workers=0)`:
  - `batch_size=8`: 8 images per batch (on Sol we use 64).
  - `shuffle=True`: a new random order **every epoch**.
    *Why?* The file is sorted by class. Without shuffling the model would see 1,257 calm images in a row, learn
    "everything is calm", then 1,268 moderate ones and forget calm. Mixed batches teach all classes at once.
  - `num_workers=0`: load images in the main program. On Sol we use 4: four helper processes read and augment
    images **while** the GPU trains on the previous batch, so the GPU never waits.
    *Concept (firmware):* like double buffering / DMA: fill the next buffer while the current one is processed.
- `iter(loader)` starts going through the loader; `next(...)` takes the **first** batch. (In training a `for` loop
  does this for every batch.)
- The DataLoader calls our `__getitem__` 8 times and **stacks** the results (lesson 2.1): images become one
  `(8, 3, 224, 224)` tensor, labels one tensor of 8 numbers.

**Expected output** (after the four lines from 2.2):

```
torch.Size([8, 3, 224, 224])
tensor([1, 1, 2, 1, 2, 2, 2, 1])
```

**Your turn**

1. Make a val dataset (`'val'`, `train=False`) and check: does `val[0]` give the same tensor every time you call
   it? (`torch.equal(val[0][0], val[0][0])`.) Does `data[0]` (train) give the same tensor twice? Why?
2. How many batches does one epoch have with 5,036 images and `batch_size=64`? (Hint: `len(loader)`.) The last
   batch isn't full; how many images does it have?
3. Set `num_workers=2` and run. It should still work, because of the `__main__` guard.

## Answers

1. `val[0]`: always the same (centre crop, no augmentation): `True`. `data[0]`: different each call (random crop,
   flip, colour): `False`. That randomness *is* the augmentation.
2. `len(loader)` = 79 (78 full batches of 64 = 4,992 images, the last batch has 44).
3. Works. (Remove the guard and try once to see what happens, then put it back.)
