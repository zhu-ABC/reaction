"""Evaluation metrics for the standalone adapter demo project."""
from __future__ import annotations

import torch


def classification_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    labels = labels.view(-1)
    return (preds == labels).float().mean().item()


def regression_mae(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    return torch.mean(torch.abs(predictions.view(-1) - labels.view(-1))).item()
