"""Sparse MoE feed-forward layer: n_experts independent copies of the same
2-layer FFN, plus a router that picks the top-k experts per token and
combines their outputs.

No capacity limit / token dropping: every token gets computed by exactly
its top-k chosen experts, however many other tokens also chose them. Keeps
the routing simple and keeps the load-balancing loss meaningful on its own
terms, since there's no capacity mechanism doing part of the balancing for
it behind the scenes.
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

        # one logit per expert per token; everything downstream (topk, softmax) is fixed math
        self.router = nn.Linear(config.d_model, config.n_experts, bias=False)

        # each expert is a smaller copy of the dense FFN (expert_d_ff < d_ff): with
        # top_k=2 of 8 experts active, active FLOPs/token stays close to the dense
        # baseline even though total params across all experts are much bigger
        self.experts = nn.ModuleList([MLP(config.d_model, config.expert_d_ff, config.dropout) for _ in range(config.n_experts)])

    def forward(self, x: torch.Tensor):
        """x: (B, T, d_model). Returns (output, router_logits), router_logits
        flattened to (B*T, n_experts) since routing is per-token anyway."""
        B, T, C = x.shape
        x_flat = x.view(-1, C)  # (N, d_model)

        router_logits = self.router(x_flat)  # (N, n_experts)
        router_probs = F.softmax(router_logits.float(), dim=-1)  # fp32 for the renorm below

        # renormalize the top-k weights to sum to 1 - otherwise combining outputs
        # with raw top-2-of-8 softmax values would just shrink the result, since
        # they rarely sum to 1 on their own
        topk_probs, topk_indices = router_probs.topk(self.top_k, dim=-1)
        topk_probs = (topk_probs / topk_probs.sum(dim=-1, keepdim=True)).to(x.dtype)

        output = torch.zeros_like(x_flat)

        # loop over experts, not tokens: gather only the tokens that picked this
        # expert, run just those, scatter-add the weighted result back
        for expert_id in range(self.n_experts):
            token_idx, slot_idx = torch.where(topk_indices == expert_id)  # slot = 1st or 2nd choice
            if token_idx.numel() == 0:
                continue

            expert_out = self.experts[expert_id](x_flat[token_idx])
            weight = topk_probs[token_idx, slot_idx].unsqueeze(-1)
            output.index_add_(0, token_idx, weight * expert_out)

        return output.view(B, T, C), router_logits


def build_ffn(config: GPTConfig):
    if config.use_moe:
        return MoEFeedForward(config)
    from .mlp import build_dense_ffn

    return build_dense_ffn(config)


def compute_aux_losses(router_logits_list: list[torch.Tensor], n_experts: int, top_k: int):
    """Switch-style load-balancing loss (generalized top-1 -> top-k) plus the
    ST-MoE router z-loss. Computed per MoE layer, then averaged - a
    well-balanced layer shouldn't be able to average out a badly imbalanced
    one. Returns (aux_loss, z_loss), zero if there are no MoE layers.

    Load-balancing: for each expert, f = fraction of tokens with it in their
    top-k (hard, no gradient) and P = average router probability on it
    (differentiable). aux_loss = n_experts * sum(f * P). Since f carries no
    gradient, minimizing this only pushes P down for experts that are
    already over-selected. At uniform routing this settles at top_k rather
    than 1 (the original paper's constant assumes top-1); what matters is
    it's still minimized at uniform routing.

    z-loss: mean(logsumexp(router_logits)^2), keeps logits from growing
    unbounded and making routing more confident/committed than it should be.
    """
    if not router_logits_list:
        zero = torch.tensor(0.0)
        return zero, zero

    aux_losses = []
    z_losses = []
    for router_logits in router_logits_list:
        router_logits = router_logits.float()
        router_probs = F.softmax(router_logits, dim=-1)

        _, topk_indices = router_probs.topk(top_k, dim=-1)
        one_hot = F.one_hot(topk_indices, n_experts).float()
        f = one_hot.sum(dim=1).mean(dim=0)
        P = router_probs.mean(dim=0)

        aux_losses.append(n_experts * (f * P).sum())
        z_losses.append(torch.logsumexp(router_logits, dim=-1).pow(2).mean())

    return torch.stack(aux_losses).mean(), torch.stack(z_losses).mean()
