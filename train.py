"""Standalone training entry for running the adapter with an MPLMM-style model.

This is a complete runnable template project. If you have the original MPLMM
model code, replace `models.mplmm.SimpleMPLMM` with the original model class and
keep the dataloader section unchanged.
"""
from __future__ import annotations

import random
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from config import TrainConfig, parse_args
from data_adapter import load_mosei_dataset, load_mosi_dataset
from models.mplmm import SimpleMPLMM
from utils.metrics import classification_accuracy, regression_mae


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_dataloaders(cfg: TrainConfig) -> tuple[DataLoader, DataLoader, DataLoader]:
    dataset_path = Path(cfg.dataset_path) if cfg.dataset_path else None
    loader_fn = load_mosi_dataset if cfg.dataset == "mosi" else load_mosei_dataset
    datasets = loader_fn(path=dataset_path, label_key=cfg.label_key)

    train_loader = DataLoader(datasets.train, batch_size=cfg.batch_size, shuffle=True)
    valid_loader = DataLoader(datasets.valid, batch_size=cfg.batch_size, shuffle=False)
    test_loader = DataLoader(datasets.test, batch_size=cfg.batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


def build_model(cfg: TrainConfig) -> nn.Module:
    num_outputs = 1 if cfg.task == "regression" else cfg.num_classes
    return SimpleMPLMM(
        text_dim=cfg.text_dim,
        audio_dim=cfg.audio_dim,
        vision_dim=cfg.vision_dim,
        hidden_dim=cfg.hidden_dim,
        num_outputs=num_outputs,
    )


def prepare_targets(batch: dict[str, torch.Tensor], cfg: TrainConfig, device: torch.device) -> torch.Tensor:
    if cfg.task == "regression":
        return batch["label"].float().to(device).view(-1, 1)
    return batch["label"].long().to(device)


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: Adam, cfg: TrainConfig, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        optimizer.zero_grad()
        logits = model(
            text=batch["text"].to(device),
            text_bert=batch["text_bert"].to(device),
            audio=batch["audio"].to(device),
            vision=batch["vision"].to(device),
        )
        targets = prepare_targets(batch, cfg, device)
        loss = criterion(logits, targets)
        loss.backward()
        optimizer.step()

        batch_size = batch["text"].size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, cfg: TrainConfig, device: torch.device) -> tuple[float, float]:
    model.eval()
    total_loss = 0.0
    total_items = 0
    predictions: list[torch.Tensor] = []
    targets_list: list[torch.Tensor] = []

    for batch in loader:
        logits = model(
            text=batch["text"].to(device),
            text_bert=batch["text_bert"].to(device),
            audio=batch["audio"].to(device),
            vision=batch["vision"].to(device),
        )
        targets = prepare_targets(batch, cfg, device)
        loss = criterion(logits, targets)

        batch_size = batch["text"].size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size
        predictions.append(logits.detach().cpu())
        targets_list.append(targets.detach().cpu())

    preds = torch.cat(predictions, dim=0)
    gold = torch.cat(targets_list, dim=0)

    if cfg.task == "regression":
        score = regression_mae(preds, gold)
    else:
        score = classification_accuracy(preds, gold)

    return total_loss / max(total_items, 1), score


def main() -> None:
    cfg = parse_args()
    if cfg.task == "regression":
        cfg.label_key = "regression_labels"

    set_seed(cfg.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader = build_dataloaders(cfg)
    model = build_model(cfg).to(device)

    criterion: nn.Module
    if cfg.task == "regression":
        criterion = nn.MSELoss()
    else:
        criterion = nn.CrossEntropyLoss()

    optimizer = Adam(model.parameters(), lr=cfg.learning_rate)

    for epoch in range(1, cfg.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, cfg, device)
        valid_loss, valid_score = evaluate(model, valid_loader, criterion, cfg, device)
        print(f"epoch={epoch} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f} valid_score={valid_score:.4f}")

    test_loss, test_score = evaluate(model, test_loader, criterion, cfg, device)
    print(f"test_loss={test_loss:.4f} test_score={test_score:.4f}")


if __name__ == "__main__":
    main()
