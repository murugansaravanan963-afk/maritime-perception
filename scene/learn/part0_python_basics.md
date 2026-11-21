# Part 0: Python basics

Eight short lessons. Each one uses something from our project, so nothing is "just an example".
Put each lesson in its own file in `scene/learn/` (`lesson01.py`, `lesson02.py`, ...) and run it with
`python scene/learn/lesson01.py` from `D:/Projects/maritime-dataset`.

---

## Lesson 0.1: Your first program

**Goal:** make the computer calculate something and show it.

```python
print("Hello, boat")

image_width = 640
image_height = 384
pixels = image_width * image_height
print(pixels)

model_name = "water segmentation"
print("Training the", model_name, "model on", pixels, "pixels per image")
```

**Line by line**

- **A program is a list of instructions** run from top to bottom, one line at a time, like a recipe.
- `print("Hello, boat")`
  - *Meaning:* show the text inside the brackets on the screen.
  - *Why:* `print` is how a program talks to you. Later we print the loss and accuracy while training.
  - The quotes `"..."` mean "this is text, not a command". Text in Python is called a **string**.
- `image_width = 640`
  - *Meaning:* make a **variable** (a labelled box) called `image_width` and put 640 in it.
  - `=` does **not** mean "equals" as in maths. It means **"put the right side into the box on the left"**.
  - *Why a variable and not just 640 everywhere?* If the size changes, you change one line. And the
    name tells the reader what 640 *is*.
  - *Concept (memory):* the computer keeps the value 640 somewhere in its memory (RAM) and the name is
    how you find it again.
- `pixels = image_width * image_height`
  - *Meaning:* open both boxes, multiply (`*`), put the answer in a new box `pixels`.
  - 640 x 384 = 245,760: the number of pixels in each of our water photos.
- `print(pixels)`: no quotes, so Python prints what is *inside* the box, not the word "pixels".
- The last `print` takes several things separated by commas and puts a space between them.
- **Names** use lower-case letters and `_` (`image_width`). Spaces are not allowed in names.

**Run:** `python scene/learn/lesson01.py`

**Expected output**

```
Hello, boat
245760
Training the water segmentation model on 245760 pixels per image
```

**Your turn**

1. Our training set has 2,312 water photos. Make a variable `train_images` and print the total number of
   pixels in all training photos. *Predict the number first.*
2. Remove the quotes around `water segmentation` and run. Read the error. What is Python complaining about?

---

## Lesson 0.2: Kinds of values, and printing numbers nicely

**Goal:** see that Python treats whole numbers, decimals, text and yes/no differently, and print a training
log line like `epoch 20  loss 1.099`.

```python
epochs = 20
learning_rate = 0.0003
loss = 1.0986122886681098
model_name = "sea state"
is_training = True

print(type(epochs), type(learning_rate), type(model_name), type(is_training))
print(epochs / 3)
print(epochs // 3, epochs % 3)
print(f"epoch {epochs}  loss {loss:.3f}  lr {learning_rate}")
print("rough" + " sea")
print(str(epochs) + " epochs")
```

**Line by line**

- `epochs = 20`: a whole number. Python calls it `int` (integer). *Epoch* = one full pass of training
  over all the training images. We will use 20 for the sea-state model.
- `learning_rate = 0.0003`: a decimal number, called `float` ("floating-point number"). The learning rate
  is how big a step the model takes each time it corrects itself (Part 3).
- `loss = 1.0986...`: also a float. *Loss* = how wrong the model is (Part 3). Smaller is better.
- `model_name = "sea state"`: text, called `str` (string).
- `is_training = True`: a yes/no value, called `bool` (boolean). Only `True` or `False`, capital first letter.
- `type(...)` tells you the kind of a value.
  - *Why care?* Each kind allows different operations. You can multiply numbers, but not text with text.
    Many beginner errors are "wrong kind of value".
- `epochs / 3` is normal division and always gives a float: 6.666...
- `epochs // 3` is **whole-number division** (drop the remainder): 6. `epochs % 3` is the **remainder**: 2.
  - *Concept (DSA):* `%` is used constantly, for example "print every 10th step": `if step % 10 == 0`.
- `f"epoch {epochs}  loss {loss:.3f}"` is an **f-string** (the `f` before the quote). Anything inside
  `{ }` is replaced by its value. `:.3f` means "as a decimal with 3 digits after the point".
  - *Why f-strings and not commas?* You control the format. A loss with 16 digits is unreadable; 3 is enough.
- `"rough" + " sea"`: `+` on text **joins** it.
- `str(epochs) + " epochs"`: `str(...)` turns the number into text so it can be joined to text.

**Expected output**

```
<class 'int'> <class 'float'> <class 'str'> <class 'bool'>
6.666666666666667
6 2
epoch 20  loss 1.099  lr 0.0003
rough sea
20 epochs
```

**Your turn**

1. Predict, then check: `7 // 2` and `7 % 2`.
2. What happens with `print("epochs: " + epochs)`? Read the error and fix it in two different ways.
3. `accuracy = 0.8734`. Print it as `accuracy 0.873`, then as `accuracy 87.3%` (hint: `:.1%`).

---

## Lesson 0.3: Lists, rows of values

**Goal:** keep our 3 sea-state class names in order, and turn a name into a number and back.

```python
classes = ["calm", "moderate", "rough"]
print(classes)
print(len(classes))
print(classes[0], classes[2])
print(classes[-1])
print(classes.index("rough"))

scores = [0.61, 0.72, 0.80]
scores.append(0.84)
print(scores)
print(scores[1:3])
print(max(scores), min(scores), round(sum(scores) / len(scores), 3))
```

**Line by line**

- `classes = ["calm", "moderate", "rough"]`: a **list**: several values in square brackets, in order.
- `len(classes)`: how many items: 3.
- `classes[0]`: the item at position 0. **Positions start at 0**, so `classes[2]` is the third item.
  - *Why start at 0?* The position is "how many steps from the start". The first item is 0 steps away.
    This comes from how memory works: item *i* is stored at *start address + i x item size*.
  - *Concept (DSA):* a list is an **array**: getting item *i* takes the same time whether the list has 3
    or 3 million items. That is called **O(1)** ("order one", constant time).
- `classes[-1]`: negative positions count from the end. `-1` is the last item.
- `classes.index("rough")`: searches the list and returns the position of `"rough"`: 2.
  - *Concept (ML):* **this is exactly how labels become numbers.** A model cannot output the word
    "rough", it outputs a number. We agree: 0 = calm, 1 = moderate, 2 = rough. `index` goes word → number;
    `classes[2]` goes number → word. **The order of this list must never change** after training,
    or every answer gets the wrong name.
  - *Concept (DSA):* `index` checks items one by one, so it takes longer on longer lists, **O(n)**. Fine for 3.
- `scores.append(0.84)`: add an item at the end. We will use this to collect results epoch by epoch.
- `scores[1:3]`: a **slice**: items from position 1 up to *but not including* 3.
- `max`, `min`, `sum` work on whole lists. `round(x, 3)` keeps 3 decimals.

**Expected output**

```
['calm', 'moderate', 'rough']
3
calm rough
rough
2
[0.61, 0.72, 0.8, 0.84]
[0.72, 0.8]
0.84 0.61 0.742
```

**Your turn**

1. What is `classes[1]`? What is `classes.index("calm")`?
2. Print the last two scores with a slice that uses a negative number.
3. What happens with `classes[3]`? Why?

---

## Lesson 0.4: Dictionaries, values with names

**Goal:** store everything we know about one image (file, split, label) together.

```python
record = {"image": "images/train/5/IMG_7783_810.png", "split": "train", "beaufort": 5, "label": "rough"}
print(record["label"])
print(record["beaufort"])

record["checked"] = True
print(record)
print(len(record))
print("split" in record, "mask" in record)
print(record.get("mask"))

records = [
    {"image": "a.png", "split": "train", "label": "calm"},
    {"image": "b.png", "split": "val", "label": "rough"},
    {"image": "c.png", "split": "train", "label": "rough"},
]
print(len(records))
print(records[1]["label"])
```

**Line by line**

- `{ "image": ..., "split": ..., ... }` is a **dictionary** (`dict`): pairs of **key: value**.
  The key is the name (`"label"`), the value is what it holds (`"rough"`).
  - *Why not a list?* With a list you must remember "position 3 is the label". With a dict you just ask
    for `"label"`. Our labels file stores every image as a dict exactly like this one.
- `record["label"]`: look up a value by its key.
  - *Concept (DSA):* a dict is a **hash map**. Python turns the key into a number (a *hash*) that points
    straight to where the value is stored, so a lookup is **O(1)**: no searching.
- `record["checked"] = True`: adding a new key is just assigning to it.
- `"split" in record`: `True` if that key exists.
- `record.get("mask")`: like `record["mask"]`, but gives `None` (Python's "nothing") instead of crashing
  when the key is missing.
- `records = [ {...}, {...}, {...} ]`: **a list of dictionaries**: one dict per image. This is the shape
  of our whole dataset.
- `records[1]["label"]`: first `[1]` picks the second image, then `["label"]` picks its label.

**Expected output**

```
rough
5
{'image': 'images/train/5/IMG_7783_810.png', 'split': 'train', 'beaufort': 5, 'label': 'rough', 'checked': True}
5
True False
None
3
rough
```

**Your turn**

1. Print the split of the third image in `records`.
2. Change the label of the first image to `"moderate"` and print the list.
3. What happens with `record["mask"]` (no `.get`)? Read the error name.

---

## Lesson 0.5: Loops and decisions

**Goal:** go through all images and count how many are rough, the core of every training script.

```python
records = [
    {"image": "a.png", "split": "train", "label": "calm"},
    {"image": "b.png", "split": "val", "label": "rough"},
    {"image": "c.png", "split": "train", "label": "rough"},
    {"image": "d.png", "split": "test", "label": "moderate"},
    {"image": "e.png", "split": "train", "label": "moderate"},
]

for record in records:
    print(record["image"], record["label"])

rough_count = 0
for record in records:
    if record["label"] == "rough":
        rough_count = rough_count + 1
print("rough images:", rough_count)

for record in records:
    if record["split"] == "train" and record["label"] != "calm":
        print("train, not calm:", record["image"])
    elif record["split"] == "val":
        print("val:", record["image"])
    else:
        print("something else:", record["image"])

for epoch in range(1, 4):
    print("epoch", epoch)
```

**Line by line**

- `for record in records:` means "**for each** item in the list, call it `record` and run the indented
  lines below". This is a **loop**.
  - The `:` at the end and the **indentation** (4 spaces) are how Python knows which lines belong to the
    loop. Other languages use `{ }`; Python uses spaces. Wrong indentation = different program.
- `rough_count = 0` then `rough_count = rough_count + 1`: a **counter**. Start at 0, add 1 each time.
  `rough_count += 1` is the short way to write the same thing.
- `if record["label"] == "rough":` runs the indented line only when the condition is true.
  - `==` **compares** (is it equal?). A single `=` **assigns** (put into the box). Mixing them up is a
    classic bug.
  - `!=` means "not equal".
- `and` means both must be true; `or` means at least one.
- `elif` = "else if": checked only when the `if` above was false. `else` catches everything left.
- `range(1, 4)` gives 1, 2, 3 (stops *before* 4, like slices). `for epoch in range(1, epochs + 1)` is
  exactly the line that runs our training epochs in Part 3.
- *Concept (DSA):* looking at every item once is a **linear scan**, **O(n)**: twice the images, twice the time.

**Expected output**

```
a.png calm
b.png rough
c.png rough
d.png moderate
e.png moderate
rough images: 2
something else: a.png
val: b.png
train, not calm: c.png
something else: d.png
train, not calm: e.png
epoch 1
epoch 2
epoch 3
```

**Your turn**

1. Count the training images (split `"train"`).
2. Print only the images that are `"val"` or `"test"`.
3. Remove the 4 spaces in front of `rough_count = rough_count + 1`. Run. What does Python say?

---

## Lesson 0.6: Functions, recipes you can reuse

**Goal:** write "count images with label X" once and use it for every class.

```python
def count_label(records, label):
    count = 0
    for record in records:
        if record["label"] == label:
            count += 1
    return count


def percent(part, whole):
    return 100 * part / whole


records = [
    {"image": "a.png", "split": "train", "label": "calm"},
    {"image": "b.png", "split": "val", "label": "rough"},
    {"image": "c.png", "split": "train", "label": "rough"},
    {"image": "d.png", "split": "test", "label": "moderate"},
    {"image": "e.png", "split": "train", "label": "moderate"},
]

for name in ["calm", "moderate", "rough"]:
    n = count_label(records, name)
    print(f"{name:9s} {n}  ({percent(n, len(records)):.0f}%)")

train = [r for r in records if r["split"] == "train"]
print(len(train), [r["image"] for r in train])
```

**Line by line**

- `def count_label(records, label):` **defines** a function called `count_label`. It needs two inputs
  (**parameters**): the list of records and which label to count.
  - *Why functions?* Without one you would copy the counting loop three times. Copies drift apart and bugs
    hide in one of them. Write it once, use it everywhere.
  - *Concept (firmware):* exactly like a C function, e.g. `int count_label(...)`.
- The indented lines are the **body**: they only run when the function is **called**.
- `return count` sends the answer back to whoever called the function and ends the function.
- `count_label(records, name)` **calls** it. Inside the function, `label` becomes whatever `name` is now.
- `percent(part, whole)` is a one-line function.
- `f"{name:9s}"`: print the text padded to 9 characters so the columns line up. `:.0f`: no decimals.
- `train = [r for r in records if r["split"] == "train"]` is a **list comprehension**: a one-line loop
  that builds a new list. Read it as: *"a list of r, for every r in records, if r's split is train"*.
  It does the same as:
  ```python
  train = []
  for r in records:
      if r["split"] == "train":
          train.append(r)
  ```
  We use this form all the time to pick the train / val / test images.
- `[r["image"] for r in train]`: a comprehension without `if`: take just the file names.

**Expected output**

```
calm      1  (20%)
moderate  2  (40%)
rough     2  (40%)
3 ['a.png', 'c.png', 'e.png']
```

**Your turn**

1. Write `count_split(records, split)` and print the count for train, val and test.
2. Write the `train = [...]` comprehension for `"val"` instead.
3. Give `percent` a default: `def percent(part, whole=5):`. What does `percent(2)` return now?

---

## Lesson 0.7: Reading our real labels file

**Goal:** open the real sea-state labels (8,014 images) and count them per split.

```python
import json
from pathlib import Path

data_dir = Path("scene/data/uniri_seastate")
labels = json.load(open(data_dir / "seastate_labels.json"))
print(labels.keys())

records = labels["records"]
print(len(records))
print(records[0])

for split in ["train", "val", "test"]:
    split_records = [r for r in records if r["split"] == split]
    rough = [r for r in split_records if r["label"] == "rough"]
    print(split, len(split_records), "rough:", len(rough))
```

**Line by line**

- `import json` loads Python's built-in JSON tool. **JSON** is a text format that stores lists and
  dictionaries in a file, exactly the kind of data from lessons 0.3-0.4. Open
  `scene/data/uniri_seastate/seastate_labels.json` in VS Code and look: it is a giant dictionary.
  - *Why JSON and not a CSV table?* Our records are dicts, some with more fields than others; JSON stores
    that directly. CSV is a flat table.
  - *Concept (firmware):* this is **serialization**: turning data in memory into bytes in a file and back.
- `from pathlib import Path` imports the `Path` tool for file locations.
  - *Why not plain text paths?* Windows writes `D:\folder\file`, Linux (Sol) writes `/folder/file`.
    `Path` handles both, so the same script runs on the laptop and on Sol.
- `data_dir / "seastate_labels.json"`: with `Path`, `/` **joins** folder and file name.
- `open(...)` opens the file; `json.load(...)` reads the whole file and turns it into Python dicts and lists.
  - *Why load it all at once?* It is only labels (a few MB). The images themselves (1.3 GB) are **not**
    loaded here; Part 2 loads them a few at a time.
- `labels.keys()`: the names in the top-level dict. `labels["records"]` is the list we want.
- The `for split in [...]` loop uses the comprehension from lesson 0.6 to split the data.

**Expected output**

```
dict_keys(['schema', 'source', 'zip_md5', 'license', 'groups', 'records'])
8014
{'image': 'images/test/4/IMG_7740_380.png', 'split': 'test', 'beaufort': 4, 'label': 'moderate', 'video': 'IMG_7740', 'frame': 380, 'bytes': 145693, 'crc32': 3079858799}
train 5036 rough: 2511
val 956 rough: 481
test 2022 rough: 972
```

The `video` field tells which ship video an image came from. Train, val and test share **no** video, so
the test really checks images the model has never seen anything like.

**Your turn**

1. Using your `count_label` function from lesson 0.6, print calm / moderate / rough counts for **train**.
2. Print `labels["groups"]`. What does it tell you about how Beaufort numbers became our 3 classes?
3. How many different videos are in the test split? (Hint: `set(...)` keeps only unique values; `len` counts them.)

---

## Lesson 0.8: Classes, your own kind of object

**Goal:** wrap "the images of one split" into one object that knows its length and gives image *i* on
request. **This is the skeleton of the PyTorch `Dataset` in Part 2.**

```python
import json
from pathlib import Path


class SeaStateList:
    def __init__(self, data_dir, split):
        self.data_dir = Path(data_dir)
        labels = json.load(open(self.data_dir / "seastate_labels.json"))
        self.records = [r for r in labels["records"] if r["split"] == split]

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        record = self.records[index]
        return record["image"], record["label"]


train = SeaStateList("scene/data/uniri_seastate", "train")
val = SeaStateList("scene/data/uniri_seastate", "val")
print(len(train), len(val))
print(train[0])
print(val[5])
print(train.data_dir)
```

**Line by line**

- `class SeaStateList:` defines a new **kind of object** (a **class**). Think of a class as a *blueprint*,
  and each object made from it (`train`, `val`) as a *house* built from that blueprint. Same blueprint,
  different contents.
  - *Why a class and not just functions?* An object keeps its **data** (the records) and the **functions
    that use that data** together. PyTorch needs our data packaged exactly like this.
- `def __init__(self, data_dir, split):` is the **setup function**, run once when an object is made.
  The double underscores mark special names Python itself calls.
- `self` = "this particular object". `self.records = ...` stores the records **inside this object**, so
  `train.records` and `val.records` are different lists.
- `def __len__(self):` is what Python calls when you write `len(train)`.
- `def __getitem__(self, index):` is what Python calls when you write `train[0]`. It returns a **tuple**:
  two values in round brackets, `(image path, label)`.
  - *Why return a pair?* Training always needs the input (image) **and** the right answer (label) together.
- `train = SeaStateList(..., "train")` builds an object: Python runs `__init__` with `split = "train"`.
- `train.data_dir`: reading a value stored on the object. (It prints with `\` on Windows: `Path` adapts.)

**Expected output**

```
5036 956
('images/train/4/IMG_7693_1680.png', 'moderate')
('images/train/4/IMG_1483_550.png', 'moderate')
scene\data\uniri_seastate
```

**Your turn**

1. Add a method `def count(self, label):` to the class that returns how many records have that label.
   Use it to print the rough count for `train` and `val`.
2. Make a third object for `"test"` and print its length.
3. Change `__getitem__` to also return the Beaufort number (three values). Print `train[0]`.

**Checkpoint:** close all files and rewrite lesson 0.8 from memory. If you can, Part 0 is done.

---

## Answers

**0.1** 1. `train_images = 2312` then `print(train_images * pixels)` → `568197120`.
2. `SyntaxError: invalid syntax`, pointing at `segmentation`. Without quotes Python reads `water` and
`segmentation` as two variable names side by side, which is not a valid sentence in Python.

**0.2** 1. `3` and `1`. 2. `TypeError: can only concatenate str (not "int") to str`. You cannot join text
and a number. Fix: `print("epochs: " + str(epochs))` or `print(f"epochs: {epochs}")`.
3. `print(f"accuracy {accuracy:.3f}")` and `print(f"accuracy {accuracy:.1%}")`.

**0.3** 1. `moderate`; `0`. 2. `print(scores[-2:])` → `[0.8, 0.84]`.
3. `IndexError: list index out of range`: there are only positions 0, 1, 2.

**0.4** 1. `print(records[2]["split"])` → `train`. 2. `records[0]["label"] = "moderate"`.
3. `KeyError: 'mask'`: that key does not exist.

**0.5** 1.
```python
train_count = 0
for record in records:
    if record["split"] == "train":
        train_count += 1
print(train_count)   # 3
```
2. `if record["split"] == "val" or record["split"] == "test": print(record["image"])`
3. `IndentationError: expected an indented block after 'if' statement`.

**0.6** 1.
```python
def count_split(records, split):
    count = 0
    for record in records:
        if record["split"] == split:
            count += 1
    return count
```
2. `val = [r for r in records if r["split"] == "val"]`. 3. `40.0`: `whole` takes the default 5.

**0.7** 1. `train = [r for r in records if r["split"] == "train"]`, then `count_label(train, "calm")` etc.
→ calm 1257, moderate 1268, rough 2511.
2. `{'1': 'calm', '2': 'calm', '3': 'moderate', '4': 'moderate', '5': 'rough', '6': 'rough', '7': 'rough', '8': 'rough'}`:
Beaufort 1-2 → calm, 3-4 → moderate, 5-8 → rough. (The keys print as text because JSON keys are always text.)
3. `print(len(set(r["video"] for r in records if r["split"] == "test")))` → `90`.

**0.8** 1.
```python
    def count(self, label):
        return len([r for r in self.records if r["label"] == label])
```
`train.count("rough")` → 2511, `val.count("rough")` → 481.
2. `test = SeaStateList("scene/data/uniri_seastate", "test")`, `len(test)` → 2022.
3. `return record["image"], record["label"], record["beaufort"]` → `('images/train/4/IMG_7693_1680.png', 'moderate', 4)`.
