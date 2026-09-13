"""One config dataclass for the whole model, base transformer + MoE fields
together. use_moe just toggles whether Block builds a dense FFN or an MoE
layer (model/block.py); nothing else needs to change when you flip it."""
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

    # --- MoE ---
    use_moe: bool = False
    n_experts: int = 8
    top_k: int = 2
    expert_d_ff: int = 512  # narrower than the dense FFN, see model/moe.py
    aux_loss_weight: float = 0.0  # load-balancing loss coefficient
    z_loss_weight: float = 0.0  # router z-loss coefficient

    def __post_init__(self):
        assert self.d_model % self.n_heads == 0, "d_model must be divisible by n_heads"
