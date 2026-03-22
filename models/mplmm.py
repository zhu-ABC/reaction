"""A small MPLMM-style baseline used to make this repository runnable.

If you have the real MPLMM implementation, swap this class out for the original
model and keep the rest of the project structure unchanged.
"""
from __future__ import annotations

import torch
from torch import nn


class SimpleMPLMM(nn.Module):
    """Minimal multimodal network with text/audio/vision fusion."""

    def __init__(self, text_dim: int, audio_dim: int, vision_dim: int, hidden_dim: int, num_outputs: int) -> None:
        super().__init__()
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.audio_proj = nn.Linear(audio_dim, hidden_dim)
        self.vision_proj = nn.Linear(vision_dim, hidden_dim)
        self.fusion = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, num_outputs),
        )

    def forward(self, text: torch.Tensor, text_bert: torch.Tensor, audio: torch.Tensor, vision: torch.Tensor) -> torch.Tensor:
        del text_bert
        text_repr = self.text_proj(text.mean(dim=1))
        audio_repr = self.audio_proj(audio.mean(dim=1))
        vision_repr = self.vision_proj(vision.mean(dim=1))
        fused = torch.cat([text_repr, audio_repr, vision_repr], dim=-1)
        return self.fusion(fused)
