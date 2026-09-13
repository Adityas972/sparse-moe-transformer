# Sparse MoE Transformer, from scratch

A small decoder-only GPT-style transformer built up incrementally:

1. **Dense baseline** -- causal self-attention with RoPE, pre-RMSNorm, dense FFN. (done)
2. **Sparse MoE layer** -- replace the FFN with 8 experts, top-2 routing. (next)
3. **Load balancing** -- Switch-style auxiliary loss + router z-loss, expert utilization logging.
4. **Ablation** -- train with/without the load-balancing loss and compare expert utilization histograms (demonstrates expert collapse).

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
