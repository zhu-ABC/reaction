"""Example MPLMM-style training entry using the local data adapter.

This file is intentionally generic because the original MPLMM repository is not
vendored in this adapter-only repository. Copy the relevant loader section into
your actual MPLMM training script and keep the rest of the original code.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from data_adapter import load_mosei_dataset, load_mosi_dataset


class DummyMPLMM(nn.Module):
    """Tiny placeholder module used only to show the expected batch interface."""

    def __init__(self, text_dim: int = 768, hidden_dim: int = 128, num_classes: int = 3) -> None:
        super().__init__()
        self.proj = nn.Linear(text_dim, hidden_dim)
        self.cls = nn.Linear(hidden_dim, num_classes)

    def forward(self, text: torch.Tensor, text_bert: torch.Tensor, audio: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        del text_bert, audio, vision
        pooled = text.mean(dim=1)
        hidden = torch.relu(self.proj(pooled))
        return self.cls(hidden)


def build_dataloaders(args: argparse.Namespace) -> tuple[DataLoader, DataLoader, DataLoader]:
    if args.dataset == "mosi":
        datasets = load_mosi_dataset(
            path=Path(args.path) if args.path else None,
            label_key=args.label_key,
        )
    else:
        datasets = load_mosei_dataset(
            path=Path(args.path) if args.path else None,
            label_key=args.label_key,
        )

    train_loader = DataLoader(datasets.train, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(datasets.valid, batch_size=args.batch_size, shuffle=False)
    test_loader = DataLoader(datasets.test, batch_size=args.batch_size, shuffle=False)
    return train_loader, valid_loader, test_loader


def train_one_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: torch.optim.Optimizer, device: torch.device) -> float:
    model.train()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        text = batch["text"].to(device)
        text_bert = batch["text_bert"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        labels = batch["label"].long().to(device)

        optimizer.zero_grad()
        logits = model(text=text, text_bert=text_bert, audio=audio, vision=vision)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        batch_size = text.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


@torch.no_grad()
def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device) -> float:
    model.eval()
    total_loss = 0.0
    total_items = 0

    for batch in loader:
        text = batch["text"].to(device)
        text_bert = batch["text_bert"].to(device)
        audio = batch["audio"].to(device)
        vision = batch["vision"].to(device)
        labels = batch["label"].long().to(device)

        logits = model(text=text, text_bert=text_bert, audio=audio, vision=vision)
        loss = criterion(logits, labels)

        batch_size = text.size(0)
        total_loss += loss.item() * batch_size
        total_items += batch_size

    return total_loss / max(total_items, 1)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Example training file showing how to use data_adapter.py with MPLMM.")
    parser.add_argument("--dataset", choices=["mosi", "mosei"], required=True)
    parser.add_argument("--path", type=str, default="", help="Optional override for the pickle path.")
    parser.add_argument("--label-key", type=str, default="classification_labels")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_loader, valid_loader, test_loader = build_dataloaders(args)

    model = DummyMPLMM().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device)
        valid_loss = evaluate(model, valid_loader, criterion, device)
        print(f"epoch={epoch} train_loss={train_loss:.4f} valid_loss={valid_loss:.4f}")

    test_loss = evaluate(model, test_loader, criterion, device)
    print(f"test_loss={test_loss:.4f}")


if __name__ == "__main__":
    main()
