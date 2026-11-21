# Learn to code by building the three scene models

You will write, from zero, the training scripts for:

1. **Sea-state model**: looks at a patch of water and says *calm*, *moderate* or *rough*.
2. **Water-segmentation model**: colours every pixel of a photo as *obstacle*, *water* or *sky*.
3. **Horizon model**: finds the line where the water ends.

No coding knowledge is assumed. Every line is explained.

## The route

| File | What you learn | You finish with |
|---|---|---|
| [part0_python_basics.md](part0_python_basics.md) | printing, variables, lists, dictionaries, loops, `if`, functions, files, classes | reading our real labels file |
| [part1_images_as_numbers.md](part1_images_as_numbers.md) | NumPy, an image is a grid of numbers, resize, flip, normalise, masks | turning one photo into what a model eats |
| [part2_pytorch_data.md](part2_pytorch_data.md) | PyTorch tensors, `Dataset`, `DataLoader`, batches | the data half of `train_seastate.py` |
| [part3_model_and_training.md](part3_model_and_training.md) | neural network, pretrained ResNet18, loss, optimiser, training loop, accuracy | **a complete training script** |
| [part4_water_segmentation.md](part4_water_segmentation.md) | per-pixel prediction, skip connections, IoU | `train_water.py` |
| [part5_horizon.md](part5_horizon.md) | predicting numbers (regression), why position matters | `train_horizon.py` |
| [part6_training_on_sol.md](part6_training_on_sol.md) | copying data, Slurm jobs, reading logs, testing once | trained models + results |

## How every lesson works

1. **Goal**: what we want the computer to do, in plain words.
2. **Code**: type it yourself. Do not copy-paste; typing is how your fingers and brain learn it.
3. **Line by line**: for each line, *what it means*, *why this and not something else*, and the *concept* behind it.
4. **Run it** and compare with the **expected output** (every expected output in these files was really produced).
5. **Your turn**: small exercises. **Try before looking at the answers** at the bottom of each file.

Rules that make you learn faster:

- **Predict before you run.** Write down what you think will print. Being wrong is where learning happens.
- **When you get an error, read it for 5-10 minutes before asking.** The last line says *what* went wrong; the line with `^^^` shows *where*.
- **Every few lessons, rewrite an earlier lesson from memory** without looking. If you can, you know it.
- Stuck for longer than 15 minutes? Paste the code and the error to Claude.

## Setup (once)

Open a terminal (in VS Code: *Terminal → New Terminal*) and go to the project folder:

```bash
cd D:/Projects/maritime-dataset
```

Every command in these lessons is run **from this folder**. Your lesson files go in `scene/learn/`.

Parts 0 and 1 need nothing new. Before Part 2, install PyTorch (CPU version, about 200 MB):

```bash
python -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

Check it worked:

```bash
python -c "import torch, torchvision; print(torch.__version__, torchvision.__version__)"
```

## If you get stuck on a script

Finished, tested versions are in `scene/learn/solutions/`. Use them only to *compare* after you tried.
Copying them teaches nothing.
