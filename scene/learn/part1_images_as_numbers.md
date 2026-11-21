# Part 1: Images are numbers

A model never "sees" a picture. It gets a grid of numbers. This part shows how a photo becomes the exact
grid our model eats. Files: `scene/learn/lesson11.py`, `lesson12.py`, ...

---

## Lesson 1.1: NumPy, lists that do maths

**Goal:** do maths on many numbers at once, and make a 2-D grid (the shape of an image and of a
confusion table).

```python
import numpy as np

scores = np.array([0.61, 0.72, 0.80, 0.84])
print(scores * 100)
print(scores.mean(), scores.max())
print(scores.shape, scores.dtype)

grid = np.zeros((3, 4), dtype=np.int64)
grid[1, 2] = 7
grid[2, 0] = 5
print(grid)
print(grid.shape)
print(grid.sum(), grid.sum(axis=0), grid.sum(axis=1))
```

**Line by line**

- `import numpy as np` loads **NumPy**, the standard library for numbers in Python. `as np` is a nickname
  so we type `np.` instead of `numpy.` (everyone uses `np`).
- `np.array([...])` turns a list into a NumPy **array**.
  - *Why not a normal list?* `[0.61, 0.72] * 100` on a list does something useless (it repeats the list
    100 times). On an array, `* 100` multiplies **every** number. And it is ~100x faster, because NumPy
    runs the loop in C instead of Python.
  - *Concept (DSA):* an array stores numbers side by side in one block of memory, all the same type. That
    is why it is fast.
- `scores.mean()`, `scores.max()`: average and largest value. (`0.7424999999999999` instead of `0.7425`:
  computers store decimals in binary, so tiny rounding errors are normal.)
- `scores.shape` = `(4,)`: 4 numbers in one row. `scores.dtype` = `float64`: the number type.
- `np.zeros((3, 4), dtype=np.int64)`: a grid of 3 rows x 4 columns, filled with 0, whole numbers.
- `grid[1, 2] = 7`: row 1, column 2 (counting from 0).
- `grid.sum(axis=0)` sums **down** each column; `axis=1` sums **across** each row.
  - *Concept (ML):* in Part 3 our **confusion table** is a grid like this: rows = true class, columns =
    predicted class. Row sums = how many images of each class exist.

**Expected output**

```
[61. 72. 80. 84.]
0.7424999999999999 0.84
(4,) float64
[[0 0 0 0]
 [0 0 7 0]
 [5 0 0 0]]
(3, 4)
12 [5 0 7 0] [0 7 5]
```

**Your turn**

1. Make `a = np.array([1, 2, 3])` and `b = np.array([10, 20, 30])`. Predict `a + b` and `a * b`.
2. Make a 3x3 grid of zeros and put 1s on the diagonal (`[0,0]`, `[1,1]`, `[2,2]`). Then try `np.eye(3)`.
3. What does `grid.max()` give? And `grid.argmax()`? (Look up what *argmax* means; we need it in Part 3.)

---

## Lesson 1.2: A photo is a grid of numbers

**Goal:** open one sea-state photo and look at its numbers.

```python
import cv2
import numpy as np
from pathlib import Path

path = Path("scene/data/uniri_seastate") / "images/train/4/IMG_7693_1680.png"
image = cv2.imread(str(path))
print(type(image))
print(image.shape, image.dtype)
print(image[0, 0])
print(image.min(), image.max())

image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
print(image[0, 0])
```

**Line by line**

- `import cv2` loads **OpenCV**, a library for reading and changing images.
  - *Why OpenCV and not another image library (Pillow)?* Both work. OpenCV gives a NumPy array directly
    and is fast; Ultralytics (your detector) uses it too.
- `cv2.imread(str(path))` reads the file into a NumPy array. `str(path)` turns the `Path` into plain text
  because OpenCV wants text.
- `image.shape` = `(331, 331, 3)`: **331 rows (height) x 331 columns (width) x 3 colours**.
  Every pixel has 3 numbers: how much blue, green and red.
- `dtype` = `uint8`: "unsigned 8-bit integer": whole numbers from **0 to 255**. 0 = no light, 255 = full.
  - *Concept (firmware):* one byte per colour, so this image is 331 x 331 x 3 = 328,683 bytes in memory.
- `image[0, 0]` = the top-left pixel: `[138 120 105]`.
- **OpenCV stores colours as B, G, R (blue first)** for historical reasons. Almost everything else,
  including the pretrained model we will use, expects **R, G, B**.
- `cv2.cvtColor(image, cv2.COLOR_BGR2RGB)` swaps the order: now `[105 120 138]`.
  - *Why it matters:* feed BGR to a model trained on RGB and red and blue are swapped. It still runs, no
    error, just worse results. Silent bugs like this are the most dangerous kind.

**Expected output**

```
<class 'numpy.ndarray'>
(331, 331, 3) uint8
[138 120 105]
17 163
[105 120 138]
```

**Your turn**

1. Print the bottom-right pixel. (Two ways: `image[330, 330]` or `image[-1, -1]`.)
2. Print `image[:, :, 0].mean()`: the average of the red channel. Then green and blue. Which is largest?
   Does that make sense for a photo of the sea?
3. Open a rough-sea image (folder `images/train/7/`, any file) and compare `image.std()` (how much the
   numbers vary) with the moderate one. Which varies more? Why might that be?

---

## Lesson 1.3: Cutting and resizing

**Goal:** make every image 224 x 224, the size our model expects, without changing the size of the waves.

```python
import cv2
import numpy as np

image = cv2.imread("scene/data/uniri_seastate/images/train/4/IMG_7693_1680.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

SIZE = 224
top = (image.shape[0] - SIZE) // 2
left = (image.shape[1] - SIZE) // 2
print(top, left)
crop = image[top:top + SIZE, left:left + SIZE]
print(crop.shape)

resized = cv2.resize(image, (SIZE, SIZE), interpolation=cv2.INTER_AREA)
print(resized.shape)

side_by_side = np.hstack([crop, resized])
cv2.imwrite("scene/learn/crop_vs_resize.png", cv2.cvtColor(side_by_side, cv2.COLOR_RGB2BGR))
print(side_by_side.shape)
```

**Line by line**

- `SIZE = 224`: capital letters = a **constant** (a value we never change). 224 x 224 is the size the
  pretrained ResNet18 was trained on.
- `top = (331 - 224) // 2 = 53`: how far down to start so the cut is centred. `//` = whole-number division.
- `image[top:top + SIZE, left:left + SIZE]` is a **2-D slice**: rows 53 to 276, columns 53 to 276.
  The result is the **centre crop**: a 224 x 224 window cut out of the photo.
  - *Concept (DSA):* slicing a NumPy array does not copy the pixels; it makes a "view" into the same memory.
    Fast, but remember it shares memory.
- `cv2.resize(image, (SIZE, SIZE))` **squeezes** the whole photo into 224 x 224 instead. Note OpenCV wants
  `(width, height)`, the opposite order of `.shape` `(height, width)`. Another classic trap.
  - `INTER_AREA` = how to compute the new pixels when shrinking (averages neighbours, looks smoothest).
- `np.hstack([...])` puts arrays **side by side** (h = horizontal). `cv2.imwrite` saves a picture so you can
  open it and look.

**Crop or resize? For sea state: crop.** Open `crop_vs_resize.png`. Resizing shrinks everything to 68%,
so the waves look smaller and calmer. **Wave size is one of the main clues for the Beaufort number**, so we
must keep it unchanged: crop. (For the water and horizon models, Part 4-5, our images are already the
right size, 640 x 384, so neither is needed.)

*Concept (ML):* every change you make to inputs must keep the **answer** the same. A change that alters
the answer (smaller waves, but label still "rough") teaches the model wrong things.

**Expected output**

```
53 53
(224, 224, 3)
(224, 224, 3)
(224, 448, 3)
```

**Your turn**

1. Cut the **top-left** 224 x 224 corner instead of the centre.
2. Pick a *random* top-left corner: `import random`, then `top = random.randint(0, 331 - 224)`. Run it a
   few times. Why could random crops be useful during training? (Answer in Part 2.)
3. Swap the two numbers in `cv2.resize(image, (300, 150))` and print the shape. Which is width?

---

## Lesson 1.4: From 0-255 to what the model expects

**Goal:** convert the pixel numbers into the scale the pretrained model was trained with, and put the
colour channels first.

```python
import cv2
import numpy as np

image = cv2.imread("scene/data/uniri_seastate/images/train/4/IMG_7693_1680.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
image = image[53:277, 53:277]

x = image.astype(np.float32) / 255.0
print(x.dtype, round(float(x.min()), 3), round(float(x.max()), 3))

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
normalized = (x - MEAN) / STD
print(normalized.shape, np.round(normalized.mean(axis=(0, 1)), 2))

chw = normalized.transpose(2, 0, 1)
print(normalized.shape, "->", chw.shape)
```

**Line by line**

- `image.astype(np.float32)` turns 0-255 whole numbers into decimals; `/ 255.0` scales them to **0.0-1.0**.
  - *Why decimals?* Training changes numbers by tiny amounts (like 0.0003). Whole numbers can't hold that.
  - *Why float32 and not float64?* Half the memory, and GPUs are built for float32. Precise enough.
- `MEAN` and `STD` are the average and spread of each colour (R, G, B) over the 1.2 million ImageNet photos
  our pretrained model learned from.
- `(x - MEAN) / STD` is **normalisation**: shift each colour so its average is about 0 and scale it so it
  spreads about 1.
  - *Why:* the pretrained model learned on numbers in this range. Give it numbers in the same range and its
    knowledge works; give it 0-255 and it is confused.
  - *How does a (224, 224, 3) grid minus a (3,) row work?* **Broadcasting**: NumPy repeats the 3 values
    for every pixel automatically. The red mean is subtracted from every red value, and so on.
- `normalized.mean(axis=(0, 1))`: average over rows and columns, one number per colour. Not exactly 0
  (this is one sea photo, not ImageNet), but in the right range.
- `transpose(2, 0, 1)` reorders the axes: from **(height, width, colour)** to **(colour, height, width)**.
  - *Why:* images are stored "colour last", but PyTorch wants "colour first": three separate 224 x 224
    sheets (red, green, blue), one behind the other. Same numbers, different order.

**Expected output**

```
float32 0.153 0.624
(224, 224, 3) [-0.44 -0.03  0.48]
(224, 224, 3) -> (3, 224, 224)
```

**Your turn**

1. Undo the normalisation: `back = normalized * STD + MEAN`. Is `back` equal to `x`? (Try
   `np.abs(back - x).max()`.)
2. What shape does `chw.transpose(1, 2, 0)` have?

---

## Lesson 1.5: Augmentation, new training images for free

**Goal:** make changed copies of an image that still have the same answer, so the model learns the
waves and not the photo's colour or lighting.

```python
import cv2
import numpy as np

image = cv2.imread("scene/data/uniri_seastate/images/train/4/IMG_7693_1680.png")
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
x = image[53:277, 53:277].astype(np.float32) / 255.0

flipped = x[:, ::-1]
grey = x.mean(axis=2, keepdims=True).repeat(3, axis=2)
print(x.shape, grey.shape)

mean = x.mean()
stronger = np.clip((x - mean) * 1.3 + mean, 0.0, 1.0)
print(round(float(x.std()), 3), round(float(stronger.std()), 3))

row = np.hstack([x, flipped, grey, stronger])
cv2.imwrite("scene/learn/augment_check.png", cv2.cvtColor((row * 255).astype(np.uint8), cv2.COLOR_RGB2BGR))
print(row.shape)
```

**Line by line**

- **Augmentation** = randomly changing training images (flip, colour, crop) while keeping the label.
  - *Concept (ML):* our 5,036 training images come from only 99 videos. Images from one video share the
    same colour and light. A model can cheat by learning "this blue-grey colour = video 7693 = moderate"
    instead of looking at the waves. That is called **overfitting**: perfect on training images, poor on
    new ones. Augmentation removes the cheat.
- `x[:, ::-1]`: `::-1` means "step backwards", so the columns are reversed: a **mirror image**.
  Waves look the same mirrored, so the label is still right.
  - *Why not flip upside down?* In a sea photo, far water (top) looks different from near water (bottom).
    Upside-down sea doesn't exist; it would teach wrong things.
- `x.mean(axis=2, keepdims=True)`: average of R, G, B for each pixel = **greyscale**. `keepdims=True`
  keeps the shape `(224, 224, 1)`; `.repeat(3, axis=2)` copies it 3 times back to `(224, 224, 3)`, because
  the model expects 3 channels.
  - *Why:* if the image sometimes has no colour, the model can't rely on colour; it must use wave shapes.
- `(x - mean) * 1.3 + mean`: **contrast**: pull pixels further from the average (×1.3 = stronger contrast,
  ×0.7 = flatter). `x.std()` (spread) goes from 0.066 to 0.085.
- `np.clip(..., 0.0, 1.0)` cuts values back into the 0-1 range so nothing goes "brighter than white".

Open `augment_check.png`: original, mirrored, grey, stronger contrast. A person would give all four the
same Beaufort number. That is the test for a good augmentation.

**Expected output**

```
(224, 224, 3) (224, 224, 3)
0.066 0.085
(224, 896, 3)
```

**Your turn**

1. Add a fifth picture with *weaker* contrast (×0.7).
2. Make a "darker" version: `x * 0.8`. Is it still the same sea state? Is darkening a good augmentation?

---

## Lesson 1.6: Masks, the answers for water segmentation

**Goal:** look at the label image of the water model: a grid where each pixel's number is its class.

```python
import cv2
import numpy as np

mask = cv2.imread("scene/data/scene_lars_v1/masks/train/yt010_15_00049.png", cv2.IMREAD_UNCHANGED)
print(mask.shape, mask.dtype)
values, counts = np.unique(mask, return_counts=True)
print(values)
print(counts)
water_fraction = (mask == 1).mean()
print(f"water covers {water_fraction:.1%} of the image")
print(mask[0, 0], mask[300, 320])

colours = np.zeros((384, 640, 3), dtype=np.uint8)
colours[mask == 0] = (255, 200, 0)
colours[mask == 1] = (0, 120, 255)
colours[mask == 2] = (180, 0, 255)
cv2.imwrite("scene/learn/mask_colours.png", cv2.cvtColor(colours, cv2.COLOR_RGB2BGR))
```

**Line by line**

- `cv2.IMREAD_UNCHANGED` reads the file **exactly as stored**: one number per pixel. Without it OpenCV
  would turn it into a 3-colour image and the class numbers could change.
- `mask.shape` = `(384, 640)`: same height and width as the photo, but **no colour axis**: each pixel holds
  one class number: **0 obstacle, 1 water, 2 sky, 255 ignore**.
- `np.unique(mask, return_counts=True)`: which values appear and how many pixels each. This canal photo has
  no sky (no 2): trees cover the top.
- `mask == 1` makes a grid of True/False ("is this pixel water?"). `.mean()` of True/False = fraction True,
  because True counts as 1 and False as 0.
- `mask[0, 0]` = 255: the grey padding we added to make every photo 640 x 384. `mask[300, 320]` = 1: water.
  - *Why an "ignore" value?* Some pixels have no right answer (padding, unclear edges). We tell the training
    "don't score these". Part 4 does that with one setting.
- `colours[mask == 1] = (0, 120, 255)`: **boolean indexing**: set every pixel where the condition is True.
  One line instead of a loop over 245,760 pixels.

**Expected output**

```
(384, 640) uint8
[  0   1 255]
[142328  88072  15360]
water covers 35.8% of the image
255 1
```

**Your turn**

1. What fraction is obstacle (`0`)? What fraction is ignore (`255`)? Do the three fractions add to 100%?
2. Open the matching photo (`scene/data/scene_lars_v1/images/train/yt010_15_00049.jpg`) and
   `mask_colours.png` side by side. Does the blue match the water?

---

## Answers

**1.1** 1. `[11 22 33]` and `[10 40 90]`. 2. `g = np.zeros((3, 3)); g[0,0] = g[1,1] = g[2,2] = 1`;
`np.eye(3)` makes the same. 3. `7`; `6`: argmax is the **position** of the largest value, counted as if the
grid were one long row (row 1 x 4 columns + column 2 = 6). In Part 3, `argmax` over the 3 class scores
gives the predicted class.

**1.2** 1. `print(image[-1, -1])`. 2. Red 96.7, green 113.2, blue 129.9: blue is largest. Sea is
blue-green, so yes. 3. Surprise: the first Beaufort-7 file (`IMG_0650_0.png`) has std 16.9, the moderate one
17.3, almost the same. So a simple hand-made rule like "more variation = rougher" does not work. That is
exactly why we train a model on thousands of examples instead.

**1.3** 1. `image[0:224, 0:224]`. 2. Each time the model sees a slightly different window of the same sea,
so it can't memorise exact pixels. 3. `(150, 300, 3)`: OpenCV takes (width, height).

**1.4** 1. Yes, up to tiny rounding (~1e-7). 2. `(224, 224, 3)`: back to the original order.

**1.5** 1. `weaker = np.clip((x - mean) * 0.7 + mean, 0.0, 1.0)`. 2. Mostly yes: the waves are the same,
only the light changed. Our final script combines contrast and brightness changes.

**1.6** 1. `(mask == 0).mean()` → 57.9%, `(mask == 255).mean()` → 6.2%; with water 35.8% they add to 100%.
