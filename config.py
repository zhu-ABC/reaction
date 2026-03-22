"""Configuration helpers for the standalone adapter demo project."""
from __future__ import annotations

import argparse
from dataclasses import dataclass


@dataclass
class TrainConfig:
    dataset: str
    dataset_path: str
    task: str
    label_key: str
    batch_size: int
    epochs: int
    learning_rate: float
    hidden_dim: int
    num_classes: int
    text_dim: int
    audio_dim: int
    vision_dim: int
    seed: int


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(description="Run MOSI/MOSEI experiments with the local MPLMM-style template.")
    parser.add_argument("--dataset", choices=["mosi", "mosei"], required=True)
    parser.add_argument("--dataset-path", default="", help="Optional pickle path override.")
    parser.add_argument("--task", choices=["classification", "regression"], default="classification")
    parser.add_argument("--label-key", default="classification_labels")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--text-dim", type=int, default=768)
    parser.add_argument("--audio-dim", type=int, default=74)
    parser.add_argument("--vision-dim", type=int, default=35)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    return TrainConfig(
        dataset=args.dataset,
        dataset_path=args.dataset_path,
        task=args.task,
        label_key=args.label_key,
        batch_size=args.batch_size,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        text_dim=args.text_dim,
        audio_dim=args.audio_dim,
        vision_dim=args.vision_dim,
        seed=args.seed,
    )
