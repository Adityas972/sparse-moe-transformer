"""Causal multi-head self-attention with RoPE applied to queries and keys."""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .rope import apply_rope, precompute_rope_cache


class CausalSelfAttention(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_heads = config.n_heads
        self.head_dim = config.d_model // config.n_heads

        # One fused linear for Q, K, V instead of three separate ones, just fewer kernel launches.
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

        # RoPE cos/sin tables, precomputed once for the max context length.
        # Registered as buffers (not parameters) so they move with .to(device)
        # and get saved/loaded with the model, but are never trained.
        cos, sin = precompute_rope_cache(config.context_length, self.head_dim, config.rope_theta)
        self.register_buffer("rope_cos", cos, persistent=False)
        self.register_buffer("rope_sin", sin, persistent=False)

        # Causal mask: position i may only attend to positions <= i.
        # Stored as a boolean buffer and reused every forward pass.
        mask = torch.tril(torch.ones(config.context_length, config.context_length, dtype=torch.bool))
        self.register_buffer("causal_mask", mask, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, T, C = x.shape  # batch, sequence length, d_model

        qkv = self.qkv_proj(x)  # (B, T, 3*C)
        q, k, v = qkv.split(C, dim=-1)  # each (B, T, C)

        # (B, T, C) -> (B, n_heads, T, head_dim) so attention is computed
        # independently per head.
        q = q.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.head_dim).transpose(1, 2)

        # Rotate q and k by position (see rope.py). v stays as-is; RoPE only needs to affect q.k scores.
        cos, sin = self.rope_cos[:T], self.rope_sin[:T]
        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Scaled dot-product attention, computed manually (rather than via
        # F.scaled_dot_product_attention) so every step is visible.
        attn_scores = q @ k.transpose(-2, -1) / math.sqrt(self.head_dim)  # (B, n_heads, T, T)
        attn_scores = attn_scores.masked_fill(~self.causal_mask[:T, :T], float("-inf"))
        attn_weights = F.softmax(attn_scores, dim=-1)
        attn_weights = self.attn_dropout(attn_weights)
        out = attn_weights @ v  # (B, n_heads, T, head_dim)

        out = out.transpose(1, 2).contiguous().view(B, T, C)  # merge heads back
        return self.resid_dropout(self.out_proj(out))
