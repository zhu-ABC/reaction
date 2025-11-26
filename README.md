# MPLMM dataset adapter

This repository provides a lightweight loader for running MPLMM with your own
CMU-MOSI/CMU-MOSEI pickle datasets. The helper functions keep the rest of the
model code untouched while exposing train/valid/test datasets ready for
`torch.utils.data.DataLoader`. By default each dataset item is returned as a
``dict`` with the same keys the original MPLMM code expects (``text``,
``text_bert``, ``audio``, ``vision``, ``label``, ``regression_label``,
``raw_text``, ``id``), so the model can stay identical.

## Paths used by default
- MOSI: `D:\data\dataset\CMU-MOSI\Processed\aligned_50.pkl`
- MOSEI: `D:\data\dataset\CMU-MOSEI\aligned_50-001.pkl`

You can override the paths by passing a different `path` argument to the helper
functions.

## Quick start
```python
from data_adapter import load_mosi_dataset, load_mosei_dataset
from torch.utils.data import DataLoader

# Load MOSI (default path). Use load_mosei_dataset for MOSEI.
datasets = load_mosi_dataset()

# Example: build loaders without touching the original MPLMM training code.
train_loader = DataLoader(datasets.train, batch_size=32, shuffle=True)
valid_loader = DataLoader(datasets.valid, batch_size=32)
test_loader = DataLoader(datasets.test, batch_size=32)

for batch in train_loader:
    # batch is a dict keyed exactly like the original MPLMM code expects
    # (text, text_bert, audio, vision, label, regression_label, raw_text, id).
    pass
```

## Where to put things inside the original MPLMM repo

If you are working inside the original `MPLMM` project, place `data_adapter.py`
at the project root (the same folder as the main MPLMM training script). A
typical layout looks like:

```
MPLMM/
├─ train.py              # or the script you normally run for MPLMM
├─ model/                # existing MPLMM model code (unchanged)
├─ utils/                # existing helpers (unchanged)
├─ data_adapter.py       # drop this file here
└─ ...
```

Once `data_adapter.py` is in place, the training script can import it directly:

```python
from data_adapter import load_mosi_dataset, load_mosei_dataset
from torch.utils.data import DataLoader

datasets = load_mosi_dataset()  # or load_mosei_dataset()
train_loader = DataLoader(datasets.train, batch_size=32, shuffle=True)
```

No other MPLMM files need to move or change. The only things you need to supply
are the pickle files themselves at the MOSI/MOSEI paths shown above (or
override `path=` when calling the loader functions).

## How to run MPLMM with your own MOSI/MOSEI pickles

The loader is designed to plug straight into the original MPLMM training code
without changing any model logic. Follow these steps:

### If you are editing the original MPLMM repo,改动集中在 `train.py`

1) **把 `data_adapter.py` 放在 MPLMM 根目录**（已完成）。

2) **打开原来的训练脚本**（通常是仓库根目录下的 `train.py` 或你自己跑的入口脚本），在“数据加载”那几行做最小改动：

```python
# 原先可能是从自带的 loader 读某个数据目录，比如
# dataset = SomeOldLoader(cfg.data_root)
# train_loader = DataLoader(dataset.train, ...)

# 改成直接用新的适配器（MOSI 或 MOSEI 二选一）：
from data_adapter import load_mosi_dataset, load_mosei_dataset
from torch.utils.data import DataLoader

# 选 MOSI：
datasets = load_mosi_dataset()  # 如需自定义路径：load_mosi_dataset(path=Path(r"E:\\...\\aligned_50.pkl"))

# 选 MOSEI：
# datasets = load_mosei_dataset()  # 如需自定义路径：load_mosei_dataset(path=Path(r"E:\\...\\aligned_50-001.pkl"))

train_loader = DataLoader(datasets.train, batch_size=32, shuffle=True)
valid_loader = DataLoader(datasets.valid, batch_size=32)
test_loader  = DataLoader(datasets.test,  batch_size=32)
```

3) **保持模型、优化器、训练循环不变**。每个 batch 仍然是 MPLMM 习惯的键：`text`、`text_bert`、`audio`、`vision`、`label`、`regression_label`、`raw_text`、`id`，所以你的 forward 与 loss 计算代码不用改，只是把新的 `train_loader` / `valid_loader` / `test_loader` 代入即可：

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

4) **只在需要时切换标签或输出类型**：
   - 做回归：`datasets = load_mosei_dataset(label_key="regression_labels")`
   - 想拿到 dataclass 而非字典：`datasets = load_mosi_dataset(return_dict=False)`

除了上述 `train.py` 中的数据加载部分，其它原 MPLMM 代码都无需增删改。

1) **Place your pickle files** at the paths you provided (or choose your own):
   - MOSI default: `D:\data\dataset\CMU-MOSI\Processed\aligned_50.pkl`
   - MOSEI default: `D:\data\dataset\CMU-MOSEI\aligned_50-001.pkl`

2) **Load the datasets** using the helper that matches your file. The helper
   returns a :class:`DatasetBundle` containing train/valid/test splits and
   emits batches as dictionaries that match MPLMM's expected keys.

```python
from pathlib import Path
from torch.utils.data import DataLoader

from data_adapter import load_mosi_dataset, load_mosei_dataset

# Use MOSI (default path). Swap to load_mosei_dataset for MOSEI.
datasets = load_mosi_dataset()

# If your file lives elsewhere, override the path explicitly:
# datasets = load_mosei_dataset(path=Path(r"E:\\mydata\\aligned_50-001.pkl"))

train_loader = DataLoader(datasets.train, batch_size=32, shuffle=True)
valid_loader = DataLoader(datasets.valid, batch_size=32)
test_loader = DataLoader(datasets.test, batch_size=32)
```

3) **Plug the loaders into your existing MPLMM training loop**. Because each
   batch is a dict with the exact field names MPLMM uses (`text`, `text_bert`,
   `audio`, `vision`, `label`, `regression_label`, `raw_text`, `id`), you can
   feed `batch["text"]`, `batch["text_bert"]`, etc., directly into the model
   without modifying any MPLMM code.

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

4) **Optional: switch to regression labels** by changing the label key if your
   MPLMM run expects continuous sentiment targets.

```python
datasets = load_mosei_dataset(label_key="regression_labels")
```

5) **Optional: receive typed samples instead of dicts** (for custom experiments)
   by setting `return_dict=False`. This keeps the model untouched while letting
   you work with the `Sample` dataclass.

```python
datasets = load_mosi_dataset(return_dict=False)
```

The loader expects each split in the pickle file to contain the fields shown in
the dataset summary you provided (`text`, `text_bert`, `audio`, `vision`,
`classification_labels`, optional `regression_labels`, `raw_text`, and `id`).
If you prefer to work with a typed dataclass instead of a ``dict`` (e.g., for
new experiments), pass ``return_dict=False`` to ``load_mosi_dataset`` or
``load_mosei_dataset`` to receive :class:`Sample` objects instead.
