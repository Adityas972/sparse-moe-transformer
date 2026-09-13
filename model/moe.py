"""
Sparse Mixture-of-Experts feed-forward layer: replaces the single dense FFN
with `n_experts` independent copies of the same 2-layer shape, plus a router
that picks the top-2 experts *per token* and combines their outputs.

No capacity limit / token dropping (see project README for why): every
token gets computed by exactly its top-2 chosen experts, however many other
tokens in the batch also chose those same experts. This keeps the routing
logic simple and, more importantly, keeps the load-balancing loss (Step 3)
meaningful on its own terms -- there's no capacity mechanism silently doing
some of the balancing work for it.

No load-balancing loss yet (that's Step 3) -- this step is purely "does
top-2 routing + weighted combination work correctly."
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import GPTConfig
from .mlp import MLP


class MoEFeedForward(nn.Module):
    def __init__(self, config: GPTConfig):
        super().__init__()
        self.n_experts = config.n_experts
        self.top_k = config.top_k

        # The router: a single linear layer producing one logit per expert
        # per token. This is *the* learned component that decides routing --
        # everything downstream of it (topk, softmax) is fixed math.
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)

        # Each expert is an independent copy of the same dense-FFN shape
        # used in Step 1, just narrower (expert_d_ff < d_ff) -- see the note
        # in GPTConfig / README on why: with n_experts=8 and top_k=2, a
        # token's *active* compute per forward pass is 2 experts, so sizing
        # each expert at roughly d_ff/2 keeps active FLOPs per token in the
        # same ballpark as the Step 1 dense FFN, while total *parameters*
        # (8 experts' worth) are much larger -- that extra capacity, rarely
        # all touched by any single token, is the whole point of MoE.
        self.experts = nn.ModuleList([MLP(config.d_model, config.expert_d_ff, config.dropout) for _ in range(config.n_experts)])

    def forward(self, x: torch.Tensor):
        """
        x: (B, T, d_model)
        Returns (output, router_logits) where router_logits is (B*T, n_experts)
        -- flattened over tokens, since Step 3's load-balancing loss and the
        expert-utilization logging both operate over "all tokens in this
        batch" rather than caring about batch/sequence position.
        """
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (N, d_model), N = B*T -- routing is per-token, batch/seq position don't matter

        router_logits = self.router(x_flat)  # (N, n_experts)
        # Softmax in float32 regardless of input dtype: router probabilities
        # feed into a renormalization (division) below, and mixed-precision
        # softmax-then-divide is a classic source of small numerical drift
        # that's cheap to just avoid.
        router_probs = F.softmax(router_logits.float(), dim=-1)  # (N, n_experts)

        # Pick each token's top-k experts and their (still full-vocab-relative)
        # softmax weights, then renormalize just those k weights to sum to 1.
        # Without renormalizing, combining outputs with the raw top-k
        # probabilities would systematically shrink the output (top-2 of 8
        # softmax values rarely sum to 1 on their own), which would look like
        # part of the model rather than an artifact of the routing math.
        topk_probs, topk_indices = router_probs.topk(self.top_k, dim=-1)  # each (N, top_k)
        topk_probs = topk_probs / topk_probs.sum(dim=-1, keepdim=True)
        topk_probs = topk_probs.to(x.dtype)

        output = torch.zeros_like(x_flat)

        # Loop over experts (not tokens): for each expert, gather only the
        # tokens that picked it, run just those through, and scatter-add the
        # weighted result back. This is the standard "dense-in-code,
        # sparse-in-compute" MoE pattern (same shape as HF's Mixtral
        # implementation) -- no expert ever sees a token that didn't select it.
        for expert_id in range(self.n_experts):
            # token_idx: which rows (tokens) chose this expert; slot_idx:
            # whether it was their 1st or 2nd choice (needed to look up the
            # right weight, since topk_probs is arranged per-token-per-slot,
            # not per-token-per-expert).
            token_idx, slot_idx = torch.where(topk_indices == expert_id)
            if token_idx.numel() == 0:
                continue  # this expert wasn't selected by anyone in this batch

            expert_out = self.experts[expert_id](x_flat[token_idx])  # (n_selected, d_model)
            weight = topk_probs[token_idx, slot_idx].unsqueeze(-1)  # (n_selected, 1)
            output.index_add_(0, token_idx, weight * expert_out)

        return output.view(B, T, C), router_logits


def build_ffn(config: GPTConfig):
    """Returns either the Step 1 dense FFN or the MoE FFN, based on config.use_moe."""
    if config.use_moe:
        return MoEFeedForward(config)
    from .mlp import build_dense_ffn

    return build_dense_ffn(config)
