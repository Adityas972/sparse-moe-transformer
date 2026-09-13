"""The full decoder-only GPT model: token embedding -> N transformer blocks
-> final norm -> output head.

No separate positional embedding table - position comes entirely from RoPE
inside attention (rope.py). The token embedding only encodes *what* token
it is.
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .block import Block
from .config import GPTConfig
from .norm import RMSNorm


class GPT(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.config = config

        self.token_emb = nn.Embedding(config.vocab_size, config.d_model)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layers)])
        self.final_norm = RMSNorm(config.d_model)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)

        # weight tying: embedding and output projection share one matrix (GPT-2 onward)
        self.lm_head.weight = self.token_emb.weight

        self.apply(self._init_weights)
        # scale down residual-sublayer output projections by 1/sqrt(2*n_layers),
        # otherwise activation variance grows with depth (GPT-2's init trick)
        for name, p in self.named_parameters():
            if name.endswith("out_proj.weight") or name.endswith("fc_out.weight"):
                nn.init.normal_(p, mean=0.0, std=0.02 / math.sqrt(2 * config.n_layers))

    def _init_weights(self, module: nn.Module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx: torch.Tensor, targets: torch.Tensor | None = None):
        """idx, targets: (B, T) token ids, targets shifted by one (or None for inference).
        Returns (logits, loss, router_logits_list) - loss is None if targets is
        None; router_logits_list has one (B*T, n_experts) tensor per MoE block,
        empty if config.use_moe is False. Used by compute_aux_losses() for the
        load-balancing/z-loss."""
        x = self.token_emb(idx)  # (B, T, d_model)
        router_logits_list = []
        for block in self.blocks:
            x, router_logits = block(x)
            if router_logits is not None:
                router_logits_list.append(router_logits)
        x = self.final_norm(x)
        logits = self.lm_head(x)  # (B, T, vocab_size)

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1))
        return logits, loss, router_logits_list

    @torch.no_grad()
    def generate(self, idx: torch.Tensor, max_new_tokens: int, temperature: float = 1.0, top_k: int | None = None):
        """Autoregressive sampling. idx: (B, T) seed tokens -> (B, T + max_new_tokens)."""
        self.eval()
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.config.context_length :]  # RoPE cache only covers context_length positions
            logits, _, _ = self(idx_cond)
            logits = logits[:, -1, :] / max(temperature, 1e-6)  # only need the next-token distribution
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            idx = torch.cat([idx, next_token], dim=1)
        self.train()
        return idx

    def num_parameters(self) -> int:
        # lm_head.weight is the same tensor as token_emb.weight (tied), so
        # named_parameters() already counts it once, not twice.
        return sum(p.numel() for p in self.parameters())
