"""
One transformer block, pre-norm style:

    x = x + Attn(RMSNorm(x))
    x = x + FFN(RMSNorm(x))

"Pre-norm" (normalize *before* the sublayer, not after) is the modern
default (GPT-2 already used it; LLaMA/PaLM/etc. all do) because it keeps a
clean residual stream -- the input to each sublayer is normalized, but the
residual addition itself is never normalized away, which makes deep stacks
much easier to train than the original post-norm Transformer.

Step 2: when config.use_moe is True, the FFN sublayer is a sparse MoE layer
instead of the dense MLP (model/moe.py). Since the MoE layer also needs to
report its router logits (for Step 3's load-balancing/z-loss and for expert
utilization logging), Block.forward now always returns a (x, router_logits)
pair -- router_logits is simply None for a dense block.
"""
import torch
import torch.nn as nn

from .attention import CausalSelfAttention
from .config import GPTConfig
from .moe import build_ffn
from .norm import RMSNorm


class Block(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model)
        self.use_moe = config.use_moe
        self.ffn = build_ffn(config)

    def forward(self, x: torch.Tensor):
        x = x + self.attn(self.attn_norm(x))
        normed = self.ffn_norm(x)
        if self.use_moe:
            ffn_out, router_logits = self.ffn(normed)
        else:
            ffn_out, router_logits = self.ffn(normed), None
        x = x + ffn_out
        return x, router_logits
