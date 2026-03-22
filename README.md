# MPLMM dataset adapter

This repository only contains the **dataset adaptation layer** for running the
original MPLMM model with your own MOSI/MOSEI-style pickle files. It does **not**
replace the original MPLMM repository. The goal is simple:

- keep the original model code unchanged,
- keep the original training loop as unchanged as possible,
- only swap the dataset-loading part.

The adapter returns each sample as a `dict` with the same keys the original
MPLMM training code usually expects:

- `text`
- `text_bert`
- `audio`
- `vision`
- `label`
- `regression_label`
- `raw_text`
- `id`

## Default dataset paths

- MOSI: `D:\data\dataset\CMU-MOSI\Processed\aligned_50.pkl`
- MOSEI: `D:\data\dataset\CMU-MOSEI\aligned_50-001.pkl`

If your pickle files are elsewhere, pass `path=...` when loading.

## What you should put into the original MPLMM project

After cloning the original MPLMM repository, place **only this file** into the
MPLMM root directory:

```text
MPLMM/
├─ train.py                 # original training entry script
├─ model/                   # original model code
├─ utils/                   # original helper code
├─ data_adapter.py          # copy this file here
└─ ...
```

That means:

- `model/` stays unchanged.
- original network definitions stay unchanged.
- loss code stays unchanged.
- optimizer code stays unchanged.
- you only change the **data-loading section** in the training script.

## The exact change you make in the original MPLMM training script

Open the script you normally run in MPLMM, usually something like `train.py`,
`main.py`, or another entry file that builds `train_loader`, `valid_loader`,
and `test_loader`.

### Step 1: add the import

```python
from pathlib import Path
from torch.utils.data import DataLoader
from data_adapter import load_mosi_dataset, load_mosei_dataset
```

### Step 2: replace the original dataset-loading block

If the original code has something conceptually like this:

```python
# old logic in MPLMM (example only)
# train_set, valid_set, test_set = build_dataset(args)
# train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
# valid_loader = DataLoader(valid_set, batch_size=args.batch_size)
# test_loader = DataLoader(test_set, batch_size=args.batch_size)
```

replace only that part with one of the following.

#### Run MOSI

```python
datasets = load_mosi_dataset()

train_loader = DataLoader(datasets.train, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(datasets.valid, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(datasets.test, batch_size=args.batch_size, shuffle=False)
```

#### Run MOSEI

```python
datasets = load_mosei_dataset()

train_loader = DataLoader(datasets.train, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(datasets.valid, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(datasets.test, batch_size=args.batch_size, shuffle=False)
```

#### If your pickle path is different

```python
datasets = load_mosi_dataset(path=Path(r"E:\\your\\path\\aligned_50.pkl"))
# or
# datasets = load_mosei_dataset(path=Path(r"E:\\your\\path\\aligned_50-001.pkl"))
```

### Step 3: keep the model/training logic the same

Your later training code can stay in the same style. Example:

```python
for batch in train_loader:
    logits = model(
        text=batch["text"],
        text_bert=batch["text_bert"],
        audio=batch["audio"],
        vision=batch["vision"],
    )
    loss = criterion(logits, batch["label"])
    loss.backward()
    optimizer.step()
```

If the original MPLMM training loop already uses batch fields with these names,
you usually do not need to change anything below the loader creation lines.

## Full example file you can compare against

See `examples/train_with_adapter.py` for a **complete training-entry example**
showing where the adapter fits in an MPLMM-style project.

## How to run after you finish editing MPLMM

Inside the original MPLMM project:

### MOSI classification

```bash
python train.py --batch_size 32
```

### MOSEI classification

If your project already has an argument like `--dataset mosei`, keep using it,
but make sure the loader branch points to `load_mosei_dataset()`.

### Regression instead of classification

If your MPLMM experiment expects regression targets, change the load line to:

```python
datasets = load_mosei_dataset(label_key="regression_labels")
```

or:

```python
datasets = load_mosi_dataset(label_key="regression_labels")
```

## Expected pickle structure

Each pickle file should be a dictionary with top-level splits:

- `train`
- `valid`
- `test`

Each split should contain these arrays/fields:

- `text`
- `text_bert`
- `audio`
- `vision`
- `classification_labels`
- optional `regression_labels`
- optional `raw_text`
- optional `id`

This matches the dataset summary you provided for MOSI/MOSEI.

## What this repository does not do

- It does not rewrite the original MPLMM model.
- It does not bundle the original MPLMM code.
- It does not guess the exact filename of the original repository's training
  script if that repository is not present here.

So if you ask "which original MPLMM file should I edit?", the practical answer
is:

1. open the script you normally run,
2. find where `train_loader` / `valid_loader` / `test_loader` are created,
3. replace only that dataset-loading block with the adapter example above.
