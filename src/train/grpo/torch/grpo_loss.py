# src/train/grpo/torch/grpo_loss.py
import torch

def grpo_loss(
    logprobs_current: torch.Tensor,
    logprobs_old: torch.Tensor,
    advantages: torch.Tensor,
    kl_divergences: torch.Tensor,
    clip_range: float = 0.2,
    kl_coeff: float = 0.1,
    reduction: str = "mean"
) -> torch.Tensor:
    """
    Computes a safer GRPO loss with ratio clamping to help avoid NaNs.

    The objective is:
        L = - E[min(r * A, clip(r, 1 - clip_range, 1 + clip_range) * A)]
            + kl_coeff * KL

    Where r = exp(logπ_new - logπ_old).

    Additional safe-guards:
    - clamp(logprobs_current - logprobs_old, -10, 10) => ratio in [exp(-10), exp(10)]
    """

    logprobs_current = logprobs_current.float()
    logprobs_old = logprobs_old.float()

    # 1) Safely clamp the difference to avoid exp overflow
    diff = torch.clamp(
        logprobs_current - logprobs_old,
        min=-10.0,
        max=10.0
    )
    ratios = torch.exp(diff)

    # 2) Standard PPO clipped surrogate
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * advantages

    # negative sign => we want to maximize => we minimize the negative
    clipped_surrogate = -torch.min(surr1, surr2)

    # 3) KL penalty
    kl_term = kl_coeff * kl_divergences

    # 4) Combine => final loss
    loss = clipped_surrogate + kl_term

    # 5) Optionally clamp the final (if needed)
    loss = torch.clamp(loss, -1e4, 1e4)  # only if extremely large

    # 6) Reduction
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        return loss