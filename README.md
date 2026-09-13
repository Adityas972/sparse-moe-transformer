# Sparse MoE Transformer, from scratch

A small GPT-style transformer I built to actually understand Mixture-of-Experts internals,
instead of just setting `num_experts=8` in someone else's config. Built in four stages, each
one trained and checked before moving to the next:

1. Normal dense decoder-only transformer (RoPE, pre-RMSNorm, causal attention).
2. Swap the FFN for 8 experts with top-2 routing. No fancy loss yet, just routing.
3. Add the load-balancing loss and router z-loss.
4. Run the actual with/without comparison and look at the histograms.

All numbers below are from real runs on my machine (Mac, MPS backend).

## Quickstart

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/prepare_data.py                     # grabs TinyShakespeare, builds char vocab

python train.py                                 # dense baseline
python generate.py --ckpt checkpoints/dense/ckpt.pt --prompt "ROMEO:"

python train.py --use_moe --aux_loss_weight 0.01  # MoE, load-balanced
python ablation.py                                 # the with/without comparison
```

## Stage 1: dense baseline

Fused QKV attention, RoPE applied to q/k inside attention (no separate position embedding
table), pre-RMSNorm blocks, weight-tied embedding/output head. Attention is written out by
hand instead of calling `F.scaled_dot_product_attention`, mostly so the mask/softmax/scaling
steps stay visible.

Tokenizer is character-level, not BPE. TinyShakespeare is small and the point of this repo is
the MoE layer, not the tokenizer, so I didn't want to pull in a subword pipeline for it.
Tradeoff is ~4x longer sequences for the same text vs BPE, with zero extra dependencies in
return. Reasoning's in `data/prepare_data.py` if you want more detail.

4.74M params, 3000 steps, ~7 minutes. Val loss bottomed at 1.49 (perplexity ~4.5) before
starting to mildly overfit, which is expected on a dataset this small. Sample output:

> ROMEO:\
> What entent'st thou must not choosing the other.
>
> GLOUCESTER:\
> What, with you? that was your worshipful witch?

## Stage 2: MoE layer, top-2 routing

`model/moe.py` replaces the dense FFN with a router (`Linear(d_model, 8)`) and 8 smaller
copies of the same FFN. Each token's top-2 experts run, weighted by renormalized router
softmax scores, and get combined. No capacity limit: every token's chosen experts always run,
no matter how many other tokens picked the same ones. Real Switch Transformer capacity-based
token dropping adds a lot of masking complexity that isn't worth it at 14M params.

Each expert is sized at half the dense FFN's width (`expert_d_ff=512`), so with 2 of 8 experts
active per token, active compute per token stays close to stage 1's dense FFN even though
total parameters are ~4x bigger.

14.19M params, same 3000 steps. Val loss came in slightly worse than the dense model (1.53 vs
1.49): more capacity, nothing telling the router to spread load, so it overfits the small
training set harder (train loss 0.87 vs 1.07). Expert usage settled into an uneven pattern
almost immediately and stayed there: `[0.09, 0.13, 0.11, 0.10, 0.11, 0.17, 0.12, 0.16]` against
an ideal of 0.125 each. Experts 5 and 7 were doing ~1.8x expert 0's work for the rest of
training.

## Stage 3: load-balancing loss + router z-loss

Two terms in `model/moe.py::compute_aux_losses`, computed per MoE layer and averaged across
layers so one balanced layer can't cover for a badly imbalanced one:

- **Load-balancing loss**, Switch Transformer's formula generalized from top-1 to top-k: for
  each expert, `f_e` (fraction of tokens that picked it, non-differentiable) times `P_e`
  (average router softmax probability on it, differentiable), summed and scaled by
  `n_experts`. `f_e` carries no gradient, so minimizing this only pushes the router's
  probability down on experts that are already over-picked.
- **Router z-loss**, from the ST-MoE paper: `mean(logsumexp(router_logits)^2)`. Stops router
  logits from growing unbounded, which would otherwise make routing more confident/committed
  than it should be and fight the balancing loss.

Checked both against theory before trusting any training run: at init, with the router
basically random, `aux_loss` came out to 2.2022 against a theoretical minimum of exactly 2.0
for top_k=2, and `z_loss` came out to 4.6755 against `log(8)^2 ≈ 4.32`. Close enough to trust
the implementation.

## Stage 4: does it actually work?

Two identical runs, same seed, same `z_loss_weight=0.001`, different `aux_loss_weight`: 0.0 vs
0.01.

<p align="center">
  <img src="assets/expert_utilization_ablation.png" alt="Expert utilization with vs without load balancing" width="850">
</p>

| | no balancing (0.0) | balanced (0.01) |
|---|---|---|
| final expert utilization | `[.085, .125, .095, .109, .109, .168, .131, .177]` | `[.130, .117, .115, .129, .131, .129, .113, .136]` |
| max-min spread | 0.092 | 0.023 (~4x tighter) |
| val perplexity | 5.57 | 5.75 |

<p align="center">
  <img src="assets/expert_utilization_spread_over_time.png" alt="Expert utilization imbalance over training" width="650">
</p>

Unbalanced imbalance shows up almost immediately and just stays for the full 3000 steps.
Balanced converges to close-to-uniform by step ~250 and holds it there.

Validation perplexity is very slightly worse with balancing (5.75 vs 5.57), and I'm not going
to pretend that's not a real result. Forcing routing toward uniform is a constraint, and on a
dataset this small (1M characters, 8 experts) there isn't enough data for the balanced
capacity to earn that constraint back in raw perplexity. The practical case for load balancing
was never "it always lowers loss" though, it's avoiding expert collapse: capacity going
permanently unused, and on real multi-GPU hardware, some devices sitting idle while others
bottleneck. With top-2 routing at this scale, "collapse" looked like a persistent ~2x skew
rather than experts dropping to near zero. Top-1 routing or a bigger model trained longer
would probably show a sharper effect.

## Other design notes

- RoPE, not learned position embeddings, applied to q/k inside attention (`model/rope.py`).
- Pre-RMSNorm (`x = x + Sublayer(RMSNorm(x))`) over the original post-norm Transformer, the
  same choice LLaMA/PaLM-style models make.
- No expert capacity limit / token dropping, covered in stage 2 above.
- Weight tying between the token embedding and LM head, standard since GPT-2.

## Layout

```
model/
  config.py     one dataclass for every hyperparameter, base model + MoE together
  rope.py       rotary position embeddings
  norm.py       RMSNorm
  attention.py  causal multi-head self-attention w/ RoPE, written by hand
  mlp.py        dense 2-layer FFN (also the shape each expert takes)
  moe.py        the MoE layer + load-balancing/z-loss math
  block.py      one transformer block (pre-norm attn + FFN/MoE)
  gpt.py        embedding -> blocks -> norm -> head, plus .generate()
data/
  prepare_data.py   downloads + tokenizes TinyShakespeare
utils/
  scheduler.py  cosine LR schedule with linear warmup
train.py        training loop, dense or MoE
generate.py     sample text from a checkpoint
ablation.py     runs both configs, plots the comparison
```

## If I keep going

- Actual capacity-based token dropping, to see real Switch-style collapse instead of the
  milder skew top-2 routing produces here.
- A real subword tokenizer, to see if the load-balancing tradeoff changes once each token
  carries more information.
- Scale up model/data size and see whether the perplexity gap in stage 4 closes or gets worse.
  Haven't run it, so no guess.
