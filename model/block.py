"""
One transformer block, pre-norm style:

    x = x + Attn(RMSNorm(x))
    x = x + FFN(RMSNorm(x))

"Pre-norm" (normalize *before* the sublayer, not after) is the modern
default (GPT-2 already used it; LLaMA/PaLM/etc. all do) because it keeps a
clean residual stream -- the input to each sublayer is normalized, but the
residual addition itself is never normalized away, which makes deep stacks
much easier to train than the original post-norm Transformer.

Step 1: FFN is always the dense MLP from mlp.py (config.use_moe is ignored
here for now). Step 2 will add the MoE branch.
"""
import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .config import GPTConfig
from .mlp import build_dense_ffn
from .norm import RMSNorm


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.ffn = build_dense_ffn(config)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x))
        x = x + self.ffn(self.ffn_norm(x))
        return x
