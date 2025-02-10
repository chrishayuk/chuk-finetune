# src/train/grpo/torch/grpo_loss.py
import numpy as np
import torch
import torch.nn.functional as F

def compute_advantages(rewards, eps=1e-8):
    """
    Normalizes the input rewards by subtracting the mean and dividing by the standard deviation.
    Accepts a Python list or a NumPy array.
    Returns a NumPy array.
    """
    # Ensure we have a NumPy array
    rewards = np.array(rewards, dtype=np.float32)

    mean = rewards.mean()
    raw_std = rewards.std()
    
    if raw_std < eps:
        return np.zeros_like(rewards, dtype=np.float32)
    else:
        return (rewards - mean) / (raw_std + eps)


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
    Computes the GRPO loss, which is similar to PPO but includes an explicit KL penalty.
    The objective is:
        L = -E[min(r * A, clip(r, 1 - clip_range, 1 + clip_range) * A)] + kl_coeff * KL

    Args:
        logprobs_current (torch.Tensor): Log probabilities under the current policy.
        logprobs_old (torch.Tensor): Log probabilities under the old policy.
        advantages (torch.Tensor): Advantage estimates for each action.
        kl_divergences (torch.Tensor): KL divergence between old and new policy distributions.
        clip_range (float): Clipping parameter (epsilon) for PPO-like objective.
        kl_coeff (float): Coefficient for the KL divergence term.
        reduction (str): Either "mean" or "sum". Specifies how to reduce the final loss.

    Returns:
        torch.Tensor: A scalar loss (if `reduction == "mean"`) or a summed loss (if `reduction == "sum"`).
    """
    # Ensure logprobs are float to avoid potential numeric issues
    logprobs_current = logprobs_current.float()
    logprobs_old = logprobs_old.float()

    # Ratio r = exp(logπ_new - logπ_old)
    ratios = torch.exp(logprobs_current - logprobs_old)

    # PPO-like clipped surrogate objective
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * advantages

    # Negative sign to turn maximization into minimization
    clipped_surrogate = -torch.min(surr1, surr2)

    # KL penalty
    kl_term = kl_coeff * kl_divergences

    # Combine terms
    loss = clipped_surrogate + kl_term

    # Reduction: mean, sum, or none
    if reduction == "mean":
        return loss.mean()
    elif reduction == "sum":
        return loss.sum()
    else:
        # Return the per-element loss without reduction
        return loss