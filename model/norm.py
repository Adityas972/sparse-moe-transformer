"""
RMSNorm: like LayerNorm but skips re-centering (no mean subtraction) and
only rescales by the root-mean-square of the activations. It's cheaper than
LayerNorm and empirically works just as well for transformers -- this is
what LLaMA, PaLM, and most modern GPT-style models use instead of LayerNorm.
"""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability regardless of the
        # input's dtype (matters if this is ever run under fp16/bf16 autocast).
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms
        return (x * self.weight).to(dtype)
