"""
Utilities to load MOSI/MOSEI style pickle datasets for MPLMM.

The loader expects the pickle file to contain a dict with ``train``, ``valid``,
``test`` splits. Each split must contain numpy arrays for audio, vision, text,
text_bert and the classification/regression labels, matching the structure the
user provided in the prompt.
"""
from __future__ import annotations

import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, MutableMapping, Optional, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor
from torch.utils.data import Dataset


SplitName = str


@dataclass
class Sample:
    """Container for a single multimodal sample."""

    text: Tensor
    text_bert: Tensor
    audio: Tensor
    vision: Tensor
    label: Tensor
    regression_label: Optional[Tensor]
    raw_text: Optional[str]
    identifier: Optional[str]


class MultimodalDataset(Dataset[Sample]):
    """Minimal dataset wrapper for MOSI/MOSEI formatted pickle files.

    The dataset can emit either the :class:`Sample` dataclass or the plain
    ``dict`` structure (matching the keys in the original MPLMM code) so you
    don't have to touch the model logic.
    """

    def __init__(
        self,
        split_data: Mapping[str, object],
        label_key: str = "classification_labels",
        regression_label_key: str = "regression_labels",
        return_dict: bool = True,
    ) -> None:
        self.split_data = split_data
        self.label_key = label_key
        self.regression_label_key = regression_label_key
        self.return_dict = return_dict

        required_arrays: Sequence[str] = ["text", "text_bert", "audio", "vision", label_key]
        missing = [key for key in required_arrays if key not in split_data]
        if missing:
            raise KeyError(f"Missing required keys in split data: {missing}")

    def __len__(self) -> int:
        return int(np.shape(self.split_data["text"])[0])

    def __getitem__(self, index: int) -> Sample:
        text = torch.as_tensor(self.split_data["text"][index], dtype=torch.float32)
        text_bert = torch.as_tensor(self.split_data["text_bert"][index], dtype=torch.long)
        audio = torch.as_tensor(self.split_data["audio"][index], dtype=torch.float32)
        vision = torch.as_tensor(self.split_data["vision"][index], dtype=torch.float32)

        label_array = self.split_data[self.label_key]
        label = torch.as_tensor(label_array[index], dtype=torch.float32)

        regression_label: Optional[Tensor] = None
        if self.regression_label_key in self.split_data:
            regression_label = torch.as_tensor(
                self.split_data[self.regression_label_key][index], dtype=torch.float32
            )

        raw_text_array: Optional[Sequence[str]] = None
        if "raw_text" in self.split_data:
            raw_text_array = self.split_data["raw_text"]
        identifier_array: Optional[Sequence[str]] = None
        if "id" in self.split_data:
            identifier_array = self.split_data["id"]

        raw_text = raw_text_array[index] if raw_text_array is not None else None
        identifier = identifier_array[index] if identifier_array is not None else None

        sample = Sample(
            text=text,
            text_bert=text_bert,
            audio=audio,
            vision=vision,
            label=label,
            regression_label=regression_label,
            raw_text=raw_text,
            identifier=identifier,
        )

        if self.return_dict:
            # Match the original MPLMM batch structure so no model code changes
            # are needed.
            return {
                "text": sample.text,
                "text_bert": sample.text_bert,
                "audio": sample.audio,
                "vision": sample.vision,
                "label": sample.label,
                "regression_label": sample.regression_label,
                "raw_text": sample.raw_text,
                "id": sample.identifier,
            }

        return sample


@dataclass
class DatasetBundle:
    """Group of train/valid/test datasets for convenience."""

    train: MultimodalDataset
    valid: MultimodalDataset
    test: MultimodalDataset


def _load_pickle(path: Path) -> MutableMapping[str, MutableMapping[str, object]]:
    with path.open("rb") as f:
        data = pickle.load(f)
    if not isinstance(data, MutableMapping):
        raise TypeError(f"Expected mapping in pickle file, got {type(data)!r}")
    return data


def load_dataset_bundle(
    dataset_path: Path,
    label_key: str = "classification_labels",
    regression_label_key: str = "regression_labels",
    return_dict: bool = True,
) -> DatasetBundle:
    """Load MOSI/MOSEI formatted pickle file and create datasets for each split."""

    if not dataset_path.exists():
        raise FileNotFoundError(f"Dataset pickle not found: {dataset_path}")

    data = _load_pickle(dataset_path)
    for split in ("train", "valid", "test"):
        if split not in data:
            raise KeyError(f"Missing split '{split}' in dataset: keys={list(data.keys())}")

    train_ds = MultimodalDataset(
        data["train"], label_key=label_key, regression_label_key=regression_label_key, return_dict=return_dict
    )
    valid_ds = MultimodalDataset(
        data["valid"], label_key=label_key, regression_label_key=regression_label_key, return_dict=return_dict
    )
    test_ds = MultimodalDataset(
        data["test"], label_key=label_key, regression_label_key=regression_label_key, return_dict=return_dict
    )
    return DatasetBundle(train=train_ds, valid=valid_ds, test=test_ds)


def load_mosi_dataset(
    label_key: str = "classification_labels",
    regression_label_key: str = "regression_labels",
    path: Optional[Path] = None,
    return_dict: bool = True,
) -> DatasetBundle:
    """Load CMU-MOSI aligned dataset.

    The default ``path`` uses the user's provided location. Override ``path`` to
    point to a different pickle file.
    """

    dataset_path = path or Path(r"D:\\data\\dataset\\CMU-MOSI\\Processed\\aligned_50.pkl")
    return load_dataset_bundle(
        dataset_path, label_key=label_key, regression_label_key=regression_label_key, return_dict=return_dict
    )


def load_mosei_dataset(
    label_key: str = "classification_labels",
    regression_label_key: str = "regression_labels",
    path: Optional[Path] = None,
    return_dict: bool = True,
) -> DatasetBundle:
    """Load CMU-MOSEI aligned dataset.

    The default ``path`` uses the user's provided location. Override ``path`` to
    point to a different pickle file.
    """

    dataset_path = path or Path(r"D:\\data\\dataset\\CMU-MOSEI\\aligned_50-001.pkl")
    return load_dataset_bundle(
        dataset_path, label_key=label_key, regression_label_key=regression_label_key, return_dict=return_dict
    )


__all__ = [
    "DatasetBundle",
    "MultimodalDataset",
    "Sample",
    "load_dataset_bundle",
    "load_mosi_dataset",
    "load_mosei_dataset",
]
