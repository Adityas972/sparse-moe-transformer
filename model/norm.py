"""RMSNorm: LayerNorm without the mean subtraction, just rescale by RMS.
Cheaper, and what LLaMA/PaLM-style models use instead of LayerNorm."""
import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # fp32 for stability, in case this ever runs under fp16/bf16 autocast
        dtype = x.dtype
        x = x.float()
        rms = torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        x = x * rms
        return (x * self.weight).to(dtype)
