"""
Step 1 training script: trains the dense (non-MoE) GPT on TinyShakespeare.

Defaults are chosen to be comfortable on a laptop CPU: context_length=128,
batch_size=64, 3000 steps finishes in a few minutes on CPU and already
produces recognizably-Shakespeare-ish character sequences. If a GPU/MPS
device is available it's used automatically (strictly faster, same numbers).

Usage:
    python data/prepare_data.py     # one-time: download + tokenize
    python train.py                 # train with defaults
    python train.py --max_steps 5000 --out_dir checkpoints/longer_run
"""
import argparse
import json
import os
import pickle
import time

import numpy as np
import torch

from model import GPT, GPTConfig
from utils.scheduler import get_lr

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")


def parse_args():
    p = argparse.ArgumentParser()
    # model shape
    p.add_argument("--n_layers", type=int, default=6)
    p.add_argument("--n_heads", type=int, default=4)
    p.add_argument("--d_model", type=int, default=256)
    p.add_argument("--d_ff", type=int, default=1024)
    p.add_argument("--context_length", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.0)
    # MoE (unused until Step 2/3 wire them into the model; accepted here
    # already so later steps -- and the Step 4 ablation script -- don't need
    # to change this file's CLI surface)
    p.add_argument("--use_moe", action="store_true")
    p.add_argument("--n_experts", type=int, default=8)
    p.add_argument("--top_k", type=int, default=2)
    p.add_argument("--expert_d_ff", type=int, default=512)
    p.add_argument("--aux_loss_weight", type=float, default=0.0)
    p.add_argument("--z_loss_weight", type=float, default=0.0)
    # training
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--warmup_steps", type=int, default=200)
    p.add_argument("--max_lr", type=float, default=3e-4)
    p.add_argument("--min_lr", type=float, default=3e-5)
    p.add_argument("--weight_decay", type=float, default=0.1)
    p.add_argument("--grad_clip", type=float, default=1.0)
    p.add_argument("--eval_interval", type=int, default=250)
    p.add_argument("--eval_iters", type=int, default=50)
    p.add_argument("--log_interval", type=int, default=50)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out_dir", type=str, default="checkpoints/dense")
    p.add_argument("--device", type=str, default=None)  # auto-detect if unset
    return p.parse_args()


def get_device(requested: str | None) -> str:
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def make_batch_getter(context_length: int, batch_size: int, device: str):
    # np.memmap avoids loading the whole (tiny, but let's be consistent with
    # how you'd do this for a real dataset) file into RAM up front; re-opened
    # each call because memmap + multiprocessing/forking doesn't always play
    # nicely if kept open across a fork -- irrelevant here (no multiprocessing)
    # but it's a one-line cost and a safe habit.
    def get_batch(split: str):
        path = os.path.join(DATA_DIR, "train.bin" if split == "train" else "val.bin")
        data = np.memmap(path, dtype=np.uint16, mode="r")
        ix = torch.randint(len(data) - context_length - 1, (batch_size,))
        x = torch.stack([torch.from_numpy(data[i : i + context_length].astype(np.int64)) for i in ix])
        y = torch.stack([torch.from_numpy(data[i + 1 : i + 1 + context_length].astype(np.int64)) for i in ix])
        return x.to(device), y.to(device)

    return get_batch


@torch.no_grad()
def estimate_loss(model, get_batch, eval_iters: int, device: str):
    model.eval()
    out = {}
    for split in ["train", "val"]:
        losses = torch.zeros(eval_iters)
        for i in range(eval_iters):
            x, y = get_batch(split)
            _, loss, _ = model(x, y)
            losses[i] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def expert_utilization(model, get_batch, n_experts: int, top_k: int, eval_iters: int = 10):
    """Step 2 sanity check (becomes the real logging target in Step 3/4):
    how many times does each expert get selected across a handful of val
    batches? A perfectly balanced router would pick each expert
    (top_k / n_experts) of the time; with no load-balancing loss yet, don't
    be surprised if this is already uneven -- that's the exact motivation
    for Step 3.
    """
    model.eval()
    counts = torch.zeros(n_experts)
    total_selections = 0
    for _ in range(eval_iters):
        x, _ = get_batch("val")
        _, _, router_logits_list = model(x)
        for router_logits in router_logits_list:  # one per MoE block
            topk = router_logits.topk(top_k, dim=-1).indices.cpu()  # (N, top_k); move to CPU before bincount/accumulation
            counts += torch.bincount(topk.flatten(), minlength=n_experts).float()
            total_selections += topk.numel()
    model.train()
    return (counts / max(total_selections, 1)).tolist()  # fraction of selections per expert


def main():
    args = parse_args()
    torch.manual_seed(args.seed)
    device = get_device(args.device)
    os.makedirs(args.out_dir, exist_ok=True)

    meta_path = os.path.join(DATA_DIR, "meta.pkl")
    if not os.path.exists(meta_path):
        raise FileNotFoundError("Run `python data/prepare_data.py` first to download and tokenize the dataset.")
    with open(meta_path, "rb") as f:
        meta = pickle.load(f)

    config = GPTConfig(
        vocab_size=meta["vocab_size"],
        context_length=args.context_length,
        n_layers=args.n_layers,
        n_heads=args.n_heads,
        d_model=args.d_model,
        d_ff=args.d_ff,
        dropout=args.dropout,
        use_moe=args.use_moe,
        n_experts=args.n_experts,
        top_k=args.top_k,
        expert_d_ff=args.expert_d_ff,
        aux_loss_weight=args.aux_loss_weight,
        z_loss_weight=args.z_loss_weight,
    )
    model = GPT(config).to(device)
    print(f"Model has {model.num_parameters() / 1e6:.2f}M parameters. Device: {device}")

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.max_lr, betas=(0.9, 0.95), weight_decay=args.weight_decay
    )
    get_batch = make_batch_getter(config.context_length, args.batch_size, device)

    log_path = os.path.join(args.out_dir, "train_log.jsonl")
    log_file = open(log_path, "w")

    t0 = time.time()
    for step in range(args.max_steps):
        lr = get_lr(step, args.warmup_steps, args.max_steps, args.max_lr, args.min_lr)
        for group in optimizer.param_groups:
            group["lr"] = lr

        x, y = get_batch("train")
        _, loss, _ = model(x, y)  # router_logits ignored for the loss until Step 3

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
        optimizer.step()

        if step % args.log_interval == 0:
            print(f"step {step:5d} | loss {loss.item():.4f} | lr {lr:.2e} | {time.time()-t0:.1f}s")

        if step % args.eval_interval == 0 or step == args.max_steps - 1:
            losses = estimate_loss(model, get_batch, args.eval_iters, device)
            val_ppl = float(np.exp(losses["val"]))
            log_entry = {
                "step": step,
                "train_loss": losses["train"],
                "val_loss": losses["val"],
                "val_ppl": val_ppl,
                "lr": lr,
            }
            eval_line = f"  eval: train_loss {losses['train']:.4f} | val_loss {losses['val']:.4f} | val_ppl {val_ppl:.2f}"
            if config.use_moe:
                util = expert_utilization(model, get_batch, config.n_experts, config.top_k)
                log_entry["expert_utilization"] = util
                eval_line += " | expert util: " + ", ".join(f"{u:.2f}" for u in util)
            print(eval_line)
            log_file.write(json.dumps(log_entry) + "\n")
            log_file.flush()

    log_file.close()

    ckpt_path = os.path.join(args.out_dir, "ckpt.pt")
    torch.save({"model_state_dict": model.state_dict(), "config": config, "meta": meta}, ckpt_path)
    print(f"Saved checkpoint to {ckpt_path}")


if __name__ == "__main__":
    main()
