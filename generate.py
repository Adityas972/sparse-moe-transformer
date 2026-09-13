"""Loads a checkpoint and samples text from it.

Usage:
    python generate.py --ckpt checkpoints/dense/ckpt.pt --prompt "ROMEO:" --max_new_tokens 300
"""
import argparse

import torch

from model import GPT


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", type=str, default="checkpoints/dense/ckpt.pt")
    p.add_argument("--prompt", type=str, default="\n")
    p.add_argument("--max_new_tokens", type=int, default=500)
    p.add_argument("--temperature", type=float, default=0.8)
    p.add_argument("--top_k", type=int, default=50)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    ckpt = torch.load(args.ckpt, map_location=args.device, weights_only=False)
    config = ckpt["config"]
    meta = ckpt["meta"]
    stoi, itos = meta["stoi"], meta["itos"]

    model = GPT(config).to(args.device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()

    encode = lambda s: [stoi[c] for c in s]
    decode = lambda ids: "".join(itos[i] for i in ids)

    idx = torch.tensor([encode(args.prompt)], dtype=torch.long, device=args.device)
    out = model.generate(idx, args.max_new_tokens, temperature=args.temperature, top_k=args.top_k)
    print(decode(out[0].tolist()))


if __name__ == "__main__":
    main()
