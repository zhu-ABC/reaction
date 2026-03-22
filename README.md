# Complete MOSI/MOSEI adapter project

This repository is now a **complete runnable project template** for testing
MOSI/MOSEI-style pickle datasets with an MPLMM-style training pipeline.

If you already have the original MPLMM repository, you can still copy only
`data_adapter.py` into that project. But if you want a full set of files to run
from directly, this repository now includes:

```text
.
├─ config.py
├─ data_adapter.py
├─ train.py
├─ requirements.txt
├─ models/
│  ├─ __init__.py
│  └─ mplmm.py
├─ utils/
│  ├─ __init__.py
│  └─ metrics.py
└─ examples/
   └─ train_with_adapter.py
```

## What each file does

- `train.py`: complete training entry script.
- `config.py`: command-line arguments and typed config object.
- `data_adapter.py`: loads your MOSI/MOSEI pickle files.
- `models/mplmm.py`: a small MPLMM-style baseline so this repository is runnable.
- `utils/metrics.py`: accuracy and MAE metrics.
- `requirements.txt`: minimal dependencies.
- `examples/train_with_adapter.py`: extra reference example.

## Your dataset paths

Default paths already match what you gave:

- MOSI: `D:\data\dataset\CMU-MOSI\Processed\aligned_50.pkl`
- MOSEI: `D:\data\dataset\CMU-MOSEI\aligned_50-001.pkl`

If needed, pass `--dataset-path` to override them.

## Install

```bash
pip install -r requirements.txt
```

## Run classification

### MOSI

```bash
python train.py --dataset mosi --task classification --batch-size 32 --epochs 5
```

### MOSEI

```bash
python train.py --dataset mosei --task classification --batch-size 32 --epochs 5
```

## Run regression

### MOSI regression

```bash
python train.py --dataset mosi --task regression --batch-size 32 --epochs 5
```

### MOSEI regression

```bash
python train.py --dataset mosei --task regression --batch-size 32 --epochs 5
```

## If your pickle path is different

```bash
python train.py --dataset mosi --dataset-path "E:\\your\\aligned_50.pkl"
python train.py --dataset mosei --dataset-path "E:\\your\\aligned_50-001.pkl"
```

## Expected pickle structure

Each pickle must be a dict with top-level keys:

- `train`
- `valid`
- `test`

Each split should contain:

- `text`
- `text_bert`
- `audio`
- `vision`
- `classification_labels`
- optional `regression_labels`
- optional `raw_text`
- optional `id`

## Important note about the model

The `models/mplmm.py` file in this repository is a **small runnable placeholder
baseline**, not the original upstream MPLMM implementation.

That is intentional because the original MPLMM source code is not included in
this repository. If you want to use the real MPLMM model:

1. keep `data_adapter.py`,
2. replace `models/mplmm.py` with the real MPLMM model code,
3. keep `train.py`'s dataloader logic,
4. adapt only the model import/constructor if needed.

## Minimal integration into the original MPLMM repository

If you only want to modify the original MPLMM project instead of using this full
template, do the following:

1. copy `data_adapter.py` into the MPLMM root directory,
2. open the script you normally run,
3. find where `train_loader`, `valid_loader`, and `test_loader` are created,
4. replace only that dataset-loading block with:

```python
from data_adapter import load_mosi_dataset, load_mosei_dataset
from torch.utils.data import DataLoader

datasets = load_mosi_dataset()  # or load_mosei_dataset()
train_loader = DataLoader(datasets.train, batch_size=args.batch_size, shuffle=True)
valid_loader = DataLoader(datasets.valid, batch_size=args.batch_size, shuffle=False)
test_loader = DataLoader(datasets.test, batch_size=args.batch_size, shuffle=False)
```

The rest of the model forward/loss code can stay in the same batch-key style:

```python
for batch in train_loader:
    logits = model(
        text=batch["text"],
        text_bert=batch["text_bert"],
        audio=batch["audio"],
        vision=batch["vision"],
    )
```
