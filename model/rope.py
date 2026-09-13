"""Rotary positional embeddings (GPT-NeoX/LLaMA style).

Instead of adding a position vector to the token embedding, RoPE rotates
each query/key vector by an angle proportional to position, with a
different rotation frequency per dimension pair. The useful property: after
rotating q at position m and k at position n, q_m . k_n depends only on
(m - n), not on m and n individually. That's why it's applied to q/k inside
attention rather than added to the embeddings once.
"""
import torch


def precompute_rope_cache(context_length: int, head_dim: int, theta: float = 10000.0):
    """cos/sin tables of shape (context_length, head_dim).

    Only head_dim/2 distinct frequencies are needed (one per dimension
    pair); each gets duplicated across both halves of head_dim so
    apply_rope is a single elementwise multiply instead of indexing pairs.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE pairing"

    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (head_dim/2,)
    positions = torch.arange(context_length).float()
    freqs = torch.outer(positions, inv_freq)  # (T, head_dim/2)
    freqs = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim), duplicated for rotate_half below
    return freqs.cos(), freqs.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """[x1, x2] -> [-x2, x1]. Combined with the duplicated cos/sin cache,
    x*cos + rotate_half(x)*sin gives the standard per-pair rotation without
    ever materializing the pairs explicitly."""
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: (B, n_heads, T, head_dim). cos/sin: (T, head_dim), broadcast over B and heads."""
    return x * cos + _rotate_half(x) * sin
