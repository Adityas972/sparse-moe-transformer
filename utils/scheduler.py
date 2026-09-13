"""Linear warmup followed by cosine decay -- the standard GPT-training LR schedule.

Warmup avoids a large, noisy first gradient step destabilizing training
before the Adam moment estimates have "warmed up". Cosine decay (rather than
a fixed LR, or step decay) smoothly reduces the LR over training, which
empirically gives lower final loss than either alternative for this kind of
model.
"""
import math


def get_lr(step: int, warmup_steps: int, max_steps: int, max_lr: float, min_lr: float) -> float:
    if step < warmup_steps:
        return max_lr * (step + 1) / warmup_steps
    if step >= max_steps:
        return min_lr
    decay_ratio = (step - warmup_steps) / (max_steps - warmup_steps)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # 1 -> 0 over the decay window
    return min_lr + coeff * (max_lr - min_lr)
