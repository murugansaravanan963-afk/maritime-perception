# Part 3: The model and the training loop

You will finish `scene/learn/train_seastate.py`: a model, a way to measure how wrong it is, a way to
improve it, and the loop that repeats that thousands of times.

---

## Lesson 3.1: What is a neural network, and why start from a trained one?

**The idea in plain words.** A neural network is a very long calculation with millions of adjustable numbers
(**weights**). Image in → a lot of multiplying and adding → 3 numbers out (one score per class). At the start the
weights are random, so the answers are random. **Training** = showing it images with known answers and nudging
every weight a tiny bit in the direction that makes the answer less wrong. Repeat a few hundred thousand times.

**ResNet18** is a well-known network design: 18 layers, 11 million weights. **Pretrained** means someone already
trained it on ImageNet (1.2 million photos, 1,000 classes: dogs, cars, boats...). Its early layers learned
general things (edges, textures, shapes) that are useful for *any* photo, including sea waves.

**Transfer learning:** take the pretrained network, replace only its last layer (1,000 ImageNet classes) with a
new one (our 3 classes), and train. With 5,036 images you could never learn edges and textures from scratch; with
transfer learning you only adapt. *That's also how your YOLOv8 detector started (from COCO weights).*

Add under the `SeaStateData` class (above `if __name__`):

```python
def build_model():
    model = torchvision.models.resnet18(weights='IMAGENET1K_V1')
    model.fc = nn.Linear(512, len(CLASSES))
    return model
```

- `torchvision.models.resnet18(weights='IMAGENET1K_V1')` builds ResNet18 and loads the ImageNet weights
  (downloaded the first time, ~45 MB, then cached).
- `model.fc` is ResNet18's last layer: a `Linear` layer from 512 numbers to 1,000 classes. We **replace** it with
  `nn.Linear(512, 3)`: 512 inputs, 3 outputs.
  - A **Linear** (fully connected) layer: each output is a weighted sum of all inputs plus a constant:
    `score_calm = w1·x1 + w2·x2 + ... + w512·x512 + b`. The new layer starts with random weights.
  - Where do the 512 come from? ResNet18 turns the image into a 7 x 7 grid of 512 features and **averages** the
    grid into 512 numbers: a summary of "what's in this picture". Our layer turns that summary into 3 scores.
- `len(CLASSES)` instead of `3`: if you ever add a class, this line adapts.

Replace the `if __name__` block with this check:

```python
if __name__ == '__main__':
    random.seed(0)
    torch.manual_seed(0)
    data = SeaStateData('scene/data/uniri_seastate', 'train', train=True)
    loader = DataLoader(data, batch_size=8, shuffle=True, num_workers=0)
    images, labels = next(iter(loader))

    model = build_model()
    outputs = model(images)
    print(outputs.shape)
    probabilities = outputs.softmax(dim=1)
    print(probabilities[0])
    print(probabilities[0].sum())
```

- `torch.manual_seed(0)`: fix PyTorch's random numbers (the new layer's starting weights, the shuffle order).
- `outputs = model(images)`: run the batch through the network. This is the **forward pass**.
- `outputs.shape` = `(8, 3)`: 8 images, 3 scores each. These raw scores are called **logits**: any size, positive
  or negative.
- `.softmax(dim=1)` turns each row of 3 scores into **3 probabilities that add up to 1**. (`dim=1` = across the 3
  classes, not across the 8 images.) Bigger score → bigger probability.
- `probabilities[0]`: image 0's probabilities. The model is untrained (random last layer), so this is a guess.

**Expected output**

```
torch.Size([8, 3])
tensor([0.2284, 0.2188, 0.5528], grad_fn=<SelectBackward0>)
tensor(1., grad_fn=<SumBackward0>)
```

`grad_fn=...` is PyTorch remembering how this number was calculated, for the improvement step (lesson 3.3).

---

## Lesson 3.2: The loss, one number for "how wrong"

Add to the check block:

```python
    loss_function = nn.CrossEntropyLoss()
    loss = loss_function(outputs, labels)
    print('loss before training', round(loss.item(), 3))
```

- **Loss** = one number that says how wrong the model is on this batch. Training = making it smaller.
- **Cross-entropy** for classification: for each image, look at the probability the model gave to the **correct**
  class, and take `-log(probability)`. Then average over the batch.
  - Correct class at probability 1.0 → loss 0. At 0.5 → 0.69. At 0.01 → 4.6. Confident and wrong is punished hard.
  - A model guessing evenly (1/3 each) gets `-log(1/3)` = **1.10**. So: loss above 1.1 = worse than guessing,
    0 = perfect. Our untrained model: 1.448 (its random guesses are a bit confident in the wrong direction).
  - *Why not simply "% wrong"?* "% wrong" jumps in steps (an image is right or not) and doesn't say *how* to change
    the weights. Cross-entropy changes smoothly with every weight, which is what the improvement step needs.
- `nn.CrossEntropyLoss` applies the softmax itself, so we give it the raw `outputs`, not the probabilities.
- `loss.item()`: turn a one-number tensor into a plain Python number.

**Expected:** `loss before training 1.448`

---

## Lesson 3.3: Learning, the improvement step

Add to the check block:

```python
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
    for step in range(1, 16):
        outputs = model(images)
        loss = loss_function(outputs, labels)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if step % 3 == 0:
            print('step', step, 'loss', round(loss.item(), 3))
    print('predicted', outputs.argmax(dim=1))
    print('true     ', labels)
```

**This is the heart of all deep learning. Five lines, always in this order:**

1. `outputs = model(images)`: **forward**: make a guess.
2. `loss = loss_function(outputs, labels)`: **measure** how wrong.
3. `optimizer.zero_grad()`: **clear** the previous step's corrections. (PyTorch *adds up* corrections by default;
   forget this line and old corrections pile up.)
4. `loss.backward()`: **backpropagation**: for every one of the 11 million weights, PyTorch works out "if this
   weight went up a tiny bit, would the loss go up or down, and how much?" That number is the weight's **gradient**.
   This is what `grad_fn` was remembering.
5. `optimizer.step()`: **update**: move every weight a small step in the direction that lowers the loss.

- `torch.optim.AdamW(...)`: the **optimiser**, the rule for how big a step each weight takes.
  - `model.parameters()`: all weights it may change.
  - `lr=3e-4` (= 0.0003): **learning rate**, the step size. Too big: the model jumps around and never settles.
    Too small: it learns very slowly. 3e-4 is a common safe value for fine-tuning with Adam.
  - `weight_decay=1e-4`: a gentle pull of all weights towards 0, which discourages memorising (overfitting).
  - *Why AdamW and not plain SGD ("stochastic gradient descent")?* Adam adapts the step size for each weight
    automatically, so it works well without much tuning. Good default for a first model.
- `outputs.argmax(dim=1)`: for each image, the position of the highest score = the predicted class.

**Expected output**

```
step 3 loss 0.018
step 6 loss 0.003
step 9 loss 0.001
step 12 loss 0.001
step 15 loss 0.0
predicted tensor([1, 0, 2, 0, 1, 2, 0, 1])
true      tensor([1, 0, 2, 0, 1, 2, 0, 1])
```

In 15 steps the model got **all 8 images right** and the loss is ~0. Did it learn sea states? **No.** It
memorised 8 pictures. That's why we never judge a model on its training images; we use **val** images it never
trained on. (This check is still useful: if a model *can't* even memorise 8 images, the code is broken.)

---

## Lesson 3.4: Measuring on val: accuracy and the confusion table

Add above `if __name__` (after `build_model`):

```python
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
```

- `@torch.no_grad()` is a **decorator** (a label on a function that changes how it runs): inside this function
  PyTorch doesn't record calculations for `backward`. We only measure, never learn, here; this saves memory and time.
- `model.eval()`: switch the model to **evaluation mode**. Some layers act differently while training
  (BatchNorm uses each batch's own average while training, the stored average when evaluating). Forgetting
  `eval()` gives wrong, unstable scores.
- `confusion`: the 3 x 3 grid from lesson 1.1. Row = true class, column = predicted class.
- `images.to(device)`: move the batch to the GPU if we have one (lesson 3.5).
- `.cpu()`: move predictions back to normal memory to count them with NumPy.
- `zip(a, b)` walks through two lists side by side: (true₀, pred₀), (true₁, pred₁), ...
- `confusion[true, pred] += 1`: one tally mark in the right cell.
- `confusion.diagonal()`: the cells where true = predicted (correct answers) for each class.
  Divided by the row sum (images of that class) = **per-class accuracy** (also called *recall*).
  `.clip(min=1)` avoids dividing by 0 if a class is missing.
- `confusion.trace()` = sum of the diagonal = all correct answers; divided by all images = **accuracy**.
  - *Why also per class?* Our train set is half rough. A model that always says "rough" gets ~50% accuracy and is
    useless. Per-class accuracy shows that immediately.

Check it on 40 val images with the untrained model:

```python
if __name__ == '__main__':
    torch.manual_seed(0)
    val = SeaStateData('scene/data/uniri_seastate', 'val', train=False, limit=40)
    loader = DataLoader(val, batch_size=8, num_workers=0)
    model = build_model()
    accuracy, per_class, confusion = evaluate(model, loader, torch.device('cpu'))
    print('accuracy', round(float(accuracy), 3))
    print('per class', np.round(per_class, 3))
    print(confusion)
```

**Expected output**

```
accuracy 0.25
per class [1. 0. 0.]
[[10  0  0]
 [ 5  0  3]
 [22  0  0]]
```

Read the table: the untrained model says "calm" for almost everything (column 0). It gets all 10 calm images
"right" by accident and every rough one wrong. That's what "not trained" looks like.

---

## Lesson 3.5: The complete script

Now replace the whole `if __name__` block with the real program: options, training loop, saving, testing.

```python
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
```

**Options (`argparse`)**

- `argparse` reads options from the command line, e.g. `python train_seastate.py --epochs 2 --limit 16`.
  - *Why not edit numbers in the file?* On Sol you start jobs with different settings without touching the code,
    and the command itself records what you ran.
- `add_argument('--epochs', type=int, default=20)`: an option named `--epochs`, converted to a whole number,
  20 if not given. `args.epochs` then holds the value.
- `action='store_true'`: an on/off switch. `--test` present → `args.test` is True.
- *Why these defaults?* 20 epochs: 5,036 images converge quickly with a pretrained model, and more epochs mostly
  memorise. Batch 64: fits easily on an A100, and bigger batches give steadier steps.

**Setup**

- `device = torch.device('cuda' if ... else 'cpu')`: a one-line `if`: use the GPU (`cuda`) when there is one.
  Same script on laptop and Sol.
- `out_dir.mkdir(parents=True, exist_ok=True)`: make the results folder (and its parents), no error if it exists.
- `build_model().to(device)`: build and move all 11 million weights onto the GPU. **The model and the data must be
  on the same device**, that's why every batch also gets `.to(device)`.

**Test mode (`if args.test:`)**

- `torch.load(out_dir / 'last.pt')` reads saved weights; `model.load_state_dict(...)` puts them into the model.
  `map_location=device` lets weights saved on a GPU load on a CPU.
- Evaluate on the **test** split, print accuracy, per-class accuracy and the confusion table, and save them to
  `test_result.json` (`json.dump` = the opposite of `json.load`).
- `dict(zip(CLASSES, ...))` builds `{'calm': 0.9, 'moderate': 0.6, ...}` from two lists.
- `return` ends `main()` here, so test mode never trains.

**Training**

- Two datasets and loaders: train (augmented, shuffled) and val (not augmented, not shuffled).
- `CosineAnnealingLR(optimizer, T_max=args.epochs)`: a **scheduler** that slowly lowers the learning rate from
  3e-4 to almost 0 over the epochs, following a cosine curve. *Why?* Big steps early to learn fast, tiny steps at
  the end to settle into a good spot instead of bouncing around it.
- `for epoch in range(1, args.epochs + 1):` one pass over all training images per epoch.
  - `model.train()`: training mode (opposite of `eval()`), needed because `evaluate()` switched it to eval mode.
  - The inner loop is lesson 3.3's five lines, for every batch.
  - `total_loss += loss.item()` then divided by the number of batches = average loss of the epoch.
  - `scheduler.step()`: lower the learning rate once per epoch.
  - `evaluate` on val, print one line per epoch.
  - `torch.save(model.state_dict(), ...)`: save the weights **every epoch**, so a job that is killed half-way
    still leaves a model behind. `state_dict()` = all weights as a dictionary {layer name: tensor}.
  - `history.json`: every epoch's numbers, to plot or put in the report.

Compare your file with `scene/learn/solutions/train_seastate.py`.

---

## Lesson 3.6: Run it on the laptop

A tiny run to prove the whole script works (16 images, 2 epochs, ~1 minute):

```bash
python scene/learn/train_seastate.py --epochs 2 --batch 8 --workers 0 --limit 16
```

```bash
python scene/learn/train_seastate.py --test --batch 8 --workers 0 --limit 16
```

Expected:

```
train images 16 val images 16 device cpu
epoch   1  loss 1.095  val accuracy 0.312  per class [1. 0. 0.]
epoch   2  loss 0.381  val accuracy 0.312  per class [1. 0. 0.]
saved runs_scene\seastate\last.pt
TEST images 16 accuracy 0.25
  calm      1.000
  moderate  0.000
  rough     0.000
confusion (rows = true, columns = predicted)
[[4 0 0]
 [5 0 0]
 [7 0 0]]
```

The loss falls (1.10 → 0.38), so learning works. The accuracy means nothing with 16 images and 2 epochs (the
model still says "calm" to everything); you're only checking that every part runs. **Then delete the `runs_scene/seastate` folder**, so this toy model
is never confused with the real one.

The real training happens on Sol (Part 6). Then do Parts 4 and 5.

**Your turn**

1. Run with `--limit 16 --epochs 1` but **without** `--workers 0`. Does it work? (4 helper processes for 16 images.)
2. Comment out `optimizer.zero_grad()` (put `#` in front) and run lesson 3.3's check again. What happens to the loss?
3. In `evaluate`, remove `model.eval()`. Run the 2-epoch training twice: once with, once without. Compare val accuracy.
4. Our train set is 50% rough. Look up `nn.CrossEntropyLoss(weight=...)`. How would you make calm and moderate
   mistakes count double? (Don't keep it; just think about why someone would.)

## Answers

1. Yes (thanks to the `__main__` guard), just slower to start.
2. The corrections of all previous steps pile up, so the steps get bigger and bigger: the loss jumps around or
   explodes instead of falling smoothly.
3. Without `eval()` BatchNorm uses each val batch's own average; scores change and are usually worse and noisier.
4. `nn.CrossEntropyLoss(weight=torch.tensor([2.0, 2.0, 1.0]).to(device))`: mistakes on calm/moderate images count
   twice, so the model can't win just by favouring the most common class.
