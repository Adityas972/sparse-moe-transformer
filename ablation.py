"""
Runs the load-balancing ablation and plots the result.

Trains two identical MoE models, same seed and same everything else, except
aux_loss_weight (0.0 vs 0.01). z_loss_weight is held constant across both so
the only variable is the load-balancing term itself.

Usage:
    python ablation.py                       # runs both trainings, then plots
    python ablation.py --skip_training       # just re-plot existing runs
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent


def run_training(out_dir: str, aux_loss_weight: float, common_args: list[str]):
    cmd = [
        sys.executable,
        str(HERE / "train.py"),
        "--use_moe",
        "--aux_loss_weight",
        str(aux_loss_weight),
        "--out_dir",
        out_dir,
        *common_args,
    ]
    print(f"\n=== {' '.join(cmd)} ===", flush=True)
    subprocess.run(cmd, check=True)


def load_log(out_dir: str) -> list[dict]:
    with open(Path(out_dir) / "train_log.jsonl") as f:
        return [json.loads(line) for line in f]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--max_steps", type=int, default=3000)
    p.add_argument("--z_loss_weight", type=float, default=0.001)
    p.add_argument("--seed", type=int, default=1337)
    p.add_argument("--out_root", type=str, default="checkpoints/ablation")
    p.add_argument("--skip_training", action="store_true", help="reuse existing runs under --out_root, just replot")
    args = p.parse_args()

    common_args = ["--max_steps", str(args.max_steps), "--z_loss_weight", str(args.z_loss_weight), "--seed", str(args.seed)]
    no_aux_dir = f"{args.out_root}_no_aux"
    with_aux_dir = f"{args.out_root}_with_aux"

    if not args.skip_training:
        run_training(no_aux_dir, 0.0, common_args)
        run_training(with_aux_dir, 0.01, common_args)

    log_no_aux = load_log(no_aux_dir)
    log_with_aux = load_log(with_aux_dir)

    util_no_aux = log_no_aux[-1]["expert_utilization"]
    util_with_aux = log_with_aux[-1]["expert_utilization"]
    n_experts = len(util_no_aux)
    ideal = 1.0 / n_experts

    # final utilization histograms, side by side
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, util, title in zip(
        axes,
        [util_no_aux, util_with_aux],
        [f"aux_loss_weight = 0.0\n(no load balancing)", f"aux_loss_weight = 0.01\n(load balanced)"],
    ):
        ax.bar(range(n_experts), util, color="#4c72b0")
        ax.axhline(ideal, color="crimson", linestyle="--", linewidth=1.2, label=f"ideal uniform ({ideal:.3f})")
        ax.set_xlabel("Expert index")
        ax.set_title(title)
        ax.set_xticks(range(n_experts))
        ax.legend(fontsize=8)
    axes[0].set_ylabel("Fraction of token-expert selections")
    fig.suptitle("Final expert utilization: with vs. without the load-balancing loss", fontsize=12)
    fig.tight_layout()
    fig.savefig("expert_utilization_ablation.png", dpi=150)
    print("Saved expert_utilization_ablation.png")

    # imbalance (max-min utilization) over training
    fig2, ax2 = plt.subplots(figsize=(7, 4.5))
    for log, label, color in [(log_no_aux, "aux_loss_weight = 0.0", "#c44e52"), (log_with_aux, "aux_loss_weight = 0.01", "#55a868")]:
        entries = [e for e in log if "expert_utilization" in e]
        steps = [e["step"] for e in entries]
        spread = [max(e["expert_utilization"]) - min(e["expert_utilization"]) for e in entries]
        ax2.plot(steps, spread, label=label, color=color, marker="o", markersize=3)
    ax2.set_xlabel("training step")
    ax2.set_ylabel("max(expert fraction) - min(expert fraction)")
    ax2.set_title("Expert utilization imbalance over training")
    ax2.legend()
    fig2.tight_layout()
    fig2.savefig("expert_utilization_spread_over_time.png", dpi=150)
    print("Saved expert_utilization_spread_over_time.png")

    print("\nFinal utilization, no aux loss:  ", [f"{u:.3f}" for u in util_no_aux])
    print("Final utilization, with aux loss:", [f"{u:.3f}" for u in util_with_aux])
    print(
        f"Max-min spread -- no_aux: {max(util_no_aux) - min(util_no_aux):.3f}  "
        f"with_aux: {max(util_with_aux) - min(util_with_aux):.3f}"
    )
    print(f"Final val_ppl -- no_aux: {log_no_aux[-1]['val_ppl']:.2f}  with_aux: {log_with_aux[-1]['val_ppl']:.2f}")


if __name__ == "__main__":
    main()
