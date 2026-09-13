"""The dense feed-forward block (used when config.use_moe=False). Also the
shape each MoE expert takes in model/moe.py, just narrower."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig


class MLP(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float = 0.0):
        super().__init__()
        self.fc_in = nn.Linear(d_model, d_ff, bias=False)
        self.fc_out = nn.Linear(d_ff, d_model, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc_out(F.gelu(self.fc_in(x))))


def build_dense_ffn(config: GPTConfig) -> MLP:
    return MLP(config.d_model, config.d_ff, config.dropout)
