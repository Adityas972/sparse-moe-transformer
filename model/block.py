"""One transformer block, pre-norm style:

    x = x + Attn(RMSNorm(x))
    x = x + FFN(RMSNorm(x))

Pre-norm (normalize before the sublayer, not after) keeps the residual
stream clean as depth increases, which is why LLaMA/PaLM-style models use
it over the original post-norm Transformer.

The FFN is either the dense MLP or an MoE layer depending on config.use_moe.
Since the MoE layer needs to report its router logits for the load-balancing
loss and utilization logging, forward() always returns (x, router_logits) -
router_logits is just None for a dense block.
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
