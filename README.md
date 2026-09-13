# Sparse MoE Transformer, from scratch

A small decoder-only GPT-style transformer built up incrementally:

1. **Dense baseline** -- causal self-attention with RoPE, pre-RMSNorm, dense FFN. (done)
2. **Sparse MoE layer** -- replace the FFN with 8 experts, top-2 routing. (done)
3. **Load balancing** -- Switch-style auxiliary loss + router z-loss, expert utilization logging. (done)
4. **Ablation** -- train with/without the load-balancing loss and compare expert utilization histograms.

## Setup

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python data/prepare_data.py   # downloads TinyShakespeare, builds char-level train/val .bin files
```

## Step 1: dense baseline

```bash
python train.py                                    # ~3000 steps, a few minutes on CPU
python generate.py --ckpt checkpoints/dense/ckpt.pt --prompt "ROMEO:"
```

### Design choices

- **Tokenizer: character-level**, not BPE. Tradeoff explained in `data/prepare_data.py`'s docstring --
  short version: zero extra dependencies/moving parts, tiny vocab, at the cost of ~4x longer sequences
  for the same text. This project is about attention/RoPE/MoE internals, not tokenization, so the
  simplest tokenizer that works was the right call.
- **RoPE, not learned positional embeddings** -- applied inside attention to q/k only (`model/rope.py`).
- **Pre-RMSNorm** -- `x = x + Sublayer(RMSNorm(x))`, the modern default (LLaMA/PaLM-style) over the
  original post-norm Transformer, because it keeps the residual stream well-behaved at depth.
- **Weight tying** between the token embedding and the output (`lm_head`) projection.
- Manual (not fused) attention implementation in `model/attention.py`, for readability over speed --
  at this scale (d_model=256, context_length=128) the difference is not noticeable.

### Project layout

```
model/
  config.py     GPTConfig dataclass (all model hyperparameters, including not-yet-used MoE fields)
  rope.py       Rotary positional embeddings
  norm.py       RMSNorm
  attention.py  Causal multi-head self-attention w/ RoPE
  mlp.py        Dense 2-layer FFN (Step 1's FFN; also the shape each MoE expert takes in Step 2)
  block.py      One transformer block (pre-norm attn + FFN)
  gpt.py        Full model: embedding -> blocks -> norm -> head, plus .generate()
data/
  prepare_data.py   Downloads + tokenizes TinyShakespeare
utils/
  scheduler.py  Cosine LR schedule with linear warmup
train.py        Training loop (CLI flags already include the Step 2/3 MoE knobs, unused until then)
generate.py     Sample text from a checkpoint
```

## Default hyperparameters (CPU-friendly)

`n_layers=6, n_heads=4, d_model=256, d_ff=1024, context_length=128, batch_size=64, max_steps=3000`
-- picked to finish a full training run in a few minutes on a laptop CPU while still producing
recognizable structure in generated text.

## Step 2: sparse MoE, top-2 routing, no load balancing yet

```bash
python train.py --use_moe --out_dir checkpoints/moe_no_aux
python generate.py --ckpt checkpoints/moe_no_aux/ckpt.pt --prompt "ROMEO:"
```

`model/moe.py` adds `MoEFeedForward`: a router (`Linear(d_model, n_experts)`) plus 8 independent
copies of the Step 1 `MLP` (each sized `expert_d_ff=512`, half the dense FFN's width -- see the
comment in `moe.py` on why: top_k=2 of 8 experts means active FFN compute per token stays roughly
matched to Step 1's dense FFN, even though total *parameters* are ~4x larger). No capacity limit --
every token is computed by its chosen top-2 experts, whichever they are, and their outputs are
combined by their (renormalized) router softmax weights.

`Block`/`GPT` now thread router logits through the model (`GPT.forward` returns
`(logits, loss, router_logits_list)`) so Step 3 can compute the load-balancing/z-loss from them --
Step 2 doesn't use them for anything yet except utilization logging in `train.py`.

**Result (3000 steps, no aux loss):** 14.19M params (vs. dense's 4.74M). Val loss bottoms at **1.53**
(ppl 4.62) around step 1250 -- slightly worse than the dense baseline's 1.49, because with no
load-balancing pressure and much more capacity, the MoE model overfits the small training set harder
(train loss reaches 0.87 vs dense's 1.07). Expert utilization stabilizes early and stays uneven for
the rest of training: `[0.09, 0.13, 0.11, 0.10, 0.11, 0.17, 0.12, 0.16]` (ideal balanced = 0.125 each)
-- experts 5 and 7 consistently get ~1.8x the traffic of expert 0. This is the baseline Step 4's
ablation will compare against.

## Step 3: load-balancing loss + router z-loss

```bash
python train.py --use_moe --aux_loss_weight 0.01 --out_dir checkpoints/moe_with_aux
```

`model/moe.py::compute_aux_losses` adds two terms, computed per MoE layer from that layer's
`router_logits` and then averaged across layers:

- **Load-balancing loss** (Switch Transformer, generalized top-1 -> top-k): for each expert `e`,
  `f_e` = fraction of tokens with `e` among their top-k picks (hard, non-differentiable), `P_e` =
  average router softmax probability on `e` (differentiable). `aux_loss = n_experts * sum_e(f_e * P_e)`.
  Gradient only flows through `P_e`, so minimizing this pushes down the router's probability on
  experts that are already over-selected. At perfect uniform routing this evaluates to `top_k` (not
  `1` as in the original top-1 paper -- the scaling constant carries over but the "balanced" value
  shifts with top_k; what matters is it's still minimized at uniform routing).
- **Router z-loss** (ST-MoE): `mean(logsumexp(router_logits)^2)`, keeps router logits from growing
  unbounded (which would otherwise make routing increasingly overconfident and fight the balancing
  loss). Used at its always-on recommended coefficient, `z_loss_weight=0.001`, in every MoE run from
  here on -- only `aux_loss_weight` is the ablation variable in Step 4.

Sanity check at initialization (random router, effectively uniform): `aux_loss=2.2022` (theoretical
minimum for top_k=2 is exactly 2.0) and `z_loss=4.6755` (`log(8)^2 ≈ 4.32` for near-zero logits over
8 experts) -- both match theory closely, confirming the formulas are implemented correctly before
looking at any training curves.

Total loss is now `cross_entropy + aux_loss_weight * aux_loss + z_loss_weight * z_loss`; all three
components are logged separately (console + `train_log.jsonl`) specifically so they can be compared
across the Step 4 ablation.

## Step 4: ablation (with vs. without load balancing)

```bash
python ablation.py                    # trains both configs (~15 min each on MPS), then plots
python ablation.py --skip_training    # just re-plot existing runs under checkpoints/ablation_*
```

Trains two MoE models identically (same seed, same `z_loss_weight=0.001`) except `aux_loss_weight`:
`0.0` vs `0.01`. Produces `expert_utilization_ablation.png` (final per-expert selection fraction,
side by side, with a dashed line at the ideal uniform `1/8`) and `expert_utilization_spread_over_time.png`
(max-min utilization gap across training, for both runs). Both are gitignored (regenerate with
`python ablation.py`) since they're fully reproducible from code + a fixed seed, same as checkpoints/logs.

### Result

| | no aux loss (0.0) | with aux loss (0.01) |
|---|---|---|
| final utilization | `[.085,.125,.095,.109,.109,.168,.131,.177]` | `[.130,.117,.115,.129,.131,.129,.113,.136]` |
| max-min spread | 0.092 | **0.023** (~4x tighter) |
| val perplexity | 5.57 | 5.75 |

The load-balancing loss does exactly what it's supposed to: expert selection goes from visibly skewed
(experts 5/7 getting ~2x expert 0's traffic, stable that way for the rest of training) to close to
uniform, and stays there from step ~250 onward (`expert_utilization_spread_over_time.png` shows both
runs converge fast, but only the balanced one converges *down* near zero rather than to a stable ~0.09
plateau).

Worth being honest about: **validation perplexity is very slightly worse with load balancing** on this
tiny dataset (5.75 vs 5.57). That's a real, expected result, not a bug -- forcing uniform routing is a
constraint that can trade a small amount of raw fit for utilization fairness, and at this scale
(1M-character dataset, 8 experts) there isn't enough data for the extra balanced capacity to pay for
itself in perplexity. The practical case for load balancing isn't "always lowers loss" -- it's avoiding
*expert collapse* (a few experts starving, useful capacity going permanently unused, and in a real
multi-GPU deployment, catastrophic compute imbalance across devices). On top-2 routing at this scale,
collapse showed up as a persistent ~2x skew rather than experts going to ~0 -- full collapse is more
dramatic with top-1 routing and/or a larger model trained longer.
