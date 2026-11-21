# Part 6: Training on Sol and reading the results

Your three scripts work on the laptop with `--limit 16`. The real training needs a GPU: Sol's A100.

---

## Lesson 6.1: The job file

Sol is shared by thousands of people. You don't run training directly; you hand Sol a **job file** that
says what you need (1 GPU, 2 hours) and what to run. A scheduler called **Slurm** starts it when a GPU is free.

Create `scene/learn/train.sbatch`:

```bash
#!/bin/bash
#SBATCH -J scene
#SBATCH -p public
#SBATCH -q class
#SBATCH -A class_eee59875463fall2026
#SBATCH -t 0-02:00:00
#SBATCH -N 1
#SBATCH -c 8
#SBATCH --mem=32G
#SBATCH -G a100:1
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err
#SBATCH --export=NONE
set -euo pipefail
module load mamba/latest
source activate maritime
cd "${SLURM_SUBMIT_DIR}"
python "$@"
```

**Line by line**

- `#!/bin/bash`: "run this file with bash", the Linux command shell.
- Lines starting with `#SBATCH` are **requests to Slurm** (for bash they are just comments):
  - `-J scene`: the job's name (shows in the queue and in log file names).
  - `-p public -q class -A class_eee59875463fall2026`: which partition, priority class and account to use: your
    class allocation, the same as your detector jobs.
  - `-t 0-02:00:00`: maximum run time 0 days 2 hours. Slurm kills the job after that. Ask for a bit more than
    you need: shorter jobs start sooner.
  - `-N 1`: one computer. `-c 8`: 8 CPU cores (for the `--workers` that load images). `--mem=32G`: 32 GB RAM.
  - `-G a100:1`: one A100 GPU.
  - `-o logs/%x-%j.out`: where printed output goes. `%x` = job name, `%j` = job number, e.g.
    `logs/scene-41234567.out`. `-e ...err`: where errors go.
  - `--export=NONE`: start with a clean environment (no leftovers from your login session).
- `set -euo pipefail`: stop at the first error instead of continuing with broken state.
- `module load mamba/latest` and `source activate maritime`: switch on the Python environment that has PyTorch,
  OpenCV and Ultralytics (the one you used for the detector).
- `cd "${SLURM_SUBMIT_DIR}"`: go to the folder you submitted from, so `scene/data/...` paths work.
- `python "$@"`: `$@` means "**all the words after the file name** when you submitted". So one job file runs
  any of our three scripts with any options.

---

## Lesson 6.2: Copy everything to Sol

On the **laptop**, from `D:/Projects/maritime-dataset` (each line asks for your ASU password / Duo):

```bash
scp transfer/scene_lars_v1.tar transfer/uniri_seastate.tar smurug25@sol.asu.edu:/scratch/smurug25/maritime-dataset/
```

```bash
scp scene/learn/train_seastate.py scene/learn/train_water.py scene/learn/train_horizon.py scene/learn/train.sbatch smurug25@sol.asu.edu:/scratch/smurug25/maritime-dataset/scene/learn/
```

The second command fails if `scene/learn` doesn't exist on Sol yet. Do step 6.3 first, then run it.

- `scp` = "secure copy": copies files to another computer over SSH. Format: `scp <what> <user>@<computer>:<folder>`.
- The two `.tar` files are the datasets (234 MB + 1.3 GB), packed as single files because copying one big
  file is much faster than 12,000 small ones.

## Lesson 6.3: Unpack and prepare (on Sol)

Log in (`ssh smurug25@sol.asu.edu`), then:

```bash
cd /scratch/smurug25/maritime-dataset
```

```bash
tar -xf scene_lars_v1.tar && tar -xf uniri_seastate.tar && mkdir -p scene/learn logs
```

```bash
ls scene/data/scene_lars_v1 scene/data/uniri_seastate
```

You should see `images masks scene_labels.json split_summary.json ...` and `images seastate_labels.json summary.json`.
(Now run the second `scp` from 6.2 on the laptop.)

The compute nodes may not reach the internet, so download the pretrained ResNet18 weights **once, on the login node**:

```bash
module load mamba/latest && source activate maritime && python -c "import torchvision; torchvision.models.resnet18(weights='IMAGENET1K_V1'); print('ok')"
```

It saves them in your home folder cache (`~/.cache/torch`), which the GPU nodes can read.

- `tar -xf` = e**x**tract **f**ile. `&&` = "run the next command only if this one worked".
- `mkdir -p` = make folders, no complaint if they exist.

---

## Lesson 6.4: Start the three trainings

```bash
sbatch scene/learn/train.sbatch scene/learn/train_seastate.py --out runs_scene/seastate
```

```bash
sbatch scene/learn/train.sbatch scene/learn/train_water.py --out runs_scene/water
```

```bash
sbatch scene/learn/train.sbatch scene/learn/train_horizon.py --out runs_scene/horizon
```

Each prints `Submitted batch job 41234567`. The three run independently, possibly at the same time.

Watch them:

```bash
squeue -u smurug25
```

- `ST` column: `PD` = pending (waiting for a GPU), `R` = running. Gone from the list = finished (or failed).

Follow one job's output live (Ctrl+C to stop watching; the job keeps running):

```bash
tail -f logs/scene-41234567.out
```

You will see one line per epoch, like on the laptop. If the job disappears quickly, read the error file:

```bash
cat logs/scene-41234567.err
```

The `.err` file also collects harmless warnings; look at the **last lines** for a real error.

## Lesson 6.5: Is it training well? Read the epoch lines

For each model, look at two things in the log (or in `runs_scene/<model>/history.json`):

1. **The loss goes down** over the epochs (quickly at first, then slowly). If it doesn't, something is broken.
2. **The val score improves, then levels off.** If the val score gets *worse* while the loss keeps going down,
   the model is memorising the training images (**overfitting**, lesson 1.5).

We do **not** pick "the best epoch" by val score. We always keep the **last** epoch (`last.pt`) with a fixed
number of epochs chosen before training. If we picked the epoch with the best val score, val would no longer be
a fair check: we would have tuned on it.

Write down the last epoch's val numbers for each model.

**About the sea-state val score.** Expect it to jump around from epoch to epoch (in laptop tests: 53% → 65% → 53%).
That's not a bug. The model tends to give **the same answer to every image of one video**, and val has only
22 videos (moderate comes from just 5). One video flipping from "moderate" to "rough" moves the score by
several percent. Neighbouring Beaufort levels (3-4 vs 5) also genuinely look alike. The test set has 90 videos,
so its number is much more trustworthy. In the report, also give the confusion table: most mistakes should be
between *neighbouring* classes (moderate ↔ rough), rarely calm ↔ rough.

## Lesson 6.6: The test, run once

The test images were never used for anything. Running the test **once**, after you're done changing things,
is what makes the final number honest. If you test, change something, and test again, the test has become
another val set. (Same rule as your detector's test set.)

```bash
sbatch scene/learn/train.sbatch scene/learn/train_seastate.py --out runs_scene/seastate --test
```

```bash
sbatch scene/learn/train.sbatch scene/learn/train_water.py --out runs_scene/water --test
```

```bash
sbatch scene/learn/train.sbatch scene/learn/train_horizon.py --out runs_scene/horizon --test
```

Each takes a minute or two. Results: the log, and `runs_scene/<model>/test_result.json`.

## Lesson 6.7: Bring results back and report them

On the **laptop**:

```bash
scp -r smurug25@sol.asu.edu:/scratch/smurug25/maritime-dataset/runs_scene .
```

(`-r` = recursive, the whole folder; `.` = "here".)

For the report, one table per model:

| Model | Data (test) | Score |
|---|---|---|
| Sea state | UNIRI-SeaState test, 2,022 patches, 90 ship videos never seen in training | accuracy, per-class accuracy, confusion table |
| Water segmentation | 293 LaRS photos from 69 recording sessions held out from training | mIoU and IoU per class |
| Horizon | 232 of those photos with a straight water edge | mean error (px at 384 height), % within 10 px |

And one sentence each on how the test set was made (from `scene/README.md`). Reviewers trust a number only when
they know nothing leaked from training into the test.

Then paste the three `test_result.json` files to Claude to go through them together, and to plan the next step:
exporting the models for the Hailo and connecting them to the detector.
