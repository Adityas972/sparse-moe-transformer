"""
Single config dataclass for the whole model. Steps 2-3 add MoE fields here;
Step 1 only uses the base fields (use_moe stays False so each Block builds a
plain dense FFN instead of an MoE layer -- see model/block.py).

Keeping every field in one place from the start means Block/GPT never need
to be rewritten later, just toggled.
"""
from dataclasses import dataclass


@dataclass
class GPTConfig:
    # --- data / tokenizer ---
    vocab_size: int = 65  # overwritten at runtime from the actual char vocab

    # --- base transformer shape ---
    context_length: int = 128
    n_layers: int = 6
    n_heads: int = 4
    d_model: int = 256
    d_ff: int = 1024  # dense FFN hidden size = 4 * d_model, the standard GPT ratio
    dropout: float = 0.0
    rope_theta: float = 10000.0  # RoPE base frequency, 10000 is the original/standard choice

    # --- MoE (Step 2+) ---
    use_moe: bool = False
    n_experts: int = 8
    top_k: int = 2
    expert_d_ff: int = 512  # each expert is smaller than the dense FFN -- see Step 2 notes
    aux_loss_weight: float = 0.0  # load-balancing loss coefficient (Step 3)
    z_loss_weight: float = 0.0  # router z-loss coefficient (Step 3)

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
