"""
Rotary Positional Embeddings (RoPE), GPT-NeoX/LLaMA style.

The core idea: instead of adding a position vector to the token embedding
(like GPT-2's learned positional embeddings), RoPE rotates each query/key
vector by an angle proportional to its position, separately for each pair of
dimensions, using a different rotation frequency per pair (slow-rotating
pairs encode coarse/long-range position, fast-rotating pairs encode
fine-grained/local position).

The payoff: after rotating q at position m and k at position n, their dot
product q_m . k_n depends only on (m - n), the *relative* distance -- not on
m and n individually. That relative-position property is why RoPE
generalizes better to sequence lengths not seen during training, compared to
absolute learned position embeddings. It's also why RoPE is applied inside
attention (to q and k) rather than added to the input embeddings once.
"""
import torch


def precompute_rope_cache(context_length: int, head_dim: int, theta: float = 10000.0):
    """Precomputes cos/sin lookup tables of shape (context_length, head_dim).

    We only need head_dim/2 distinct frequencies (one per dimension *pair*),
    then duplicate each frequency across both halves of head_dim so that
    apply_rope can do a single elementwise multiply against a full-width
    tensor instead of indexing pairs at call time.
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE pairing"

    # theta_i = base^(-2i/head_dim) for i in [0, head_dim/2) -- i=0 rotates
    # slowest per dimension pair... wait, actually i=0 gives theta_0=1 (fastest
    # rotation, changes every position), and larger i gives smaller theta_i
    # (slower rotation). This spread of frequencies is what lets RoPE encode
    # position at multiple "resolutions" simultaneously.
    inv_freq = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))  # (head_dim/2,)

    positions = torch.arange(context_length).float()  # (T,)
    freqs = torch.outer(positions, inv_freq)  # (T, head_dim/2), freqs[t, i] = t * theta_i

    # Duplicate each frequency into both halves: [f0, f1, ..., f0, f1, ...]
    # so cos/sin end up shape (T, head_dim), matching the "rotate_half" trick below.
    freqs = torch.cat([freqs, freqs], dim=-1)  # (T, head_dim)
    return freqs.cos(), freqs.sin()


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    """Splits the last dim in half [x1, x2] and returns [-x2, x1].

    Combined with the duplicated cos/sin cache above, this implements the
    2D-rotation-per-pair math without ever materializing pairs explicitly:
    for each original pair (x1_i, x2_i), the standard rotation
        x1' = x1*cos - x2*sin
        x2' = x2*cos + x1*sin
    falls out of  x*cos + rotate_half(x)*sin  once cos/sin are duplicated
    across both halves.
    """
    d = x.shape[-1]
    x1, x2 = x[..., : d // 2], x[..., d // 2 :]
    return torch.cat([-x2, x1], dim=-1)


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Applies RoPE to a tensor of shape (B, n_heads, T, head_dim).

    cos/sin come from precompute_rope_cache and have shape (T, head_dim);
    they broadcast over the batch and head dimensions.
    """
    return x * cos + _rotate_half(x) * sin
