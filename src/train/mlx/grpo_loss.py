# src/train/mlx/grpo_loss.py
import numpy as np
import mlx.core as mx

def compute_advantages(rewards):
    # get the rewards
    rewards = np.array(rewards, dtype=np.float32)

    # get the mean
    mean = rewards.mean()

    # get standard deviation
    raw_std = rewards.std()
    
    if raw_std < 1e-8:
        return np.zeros_like(rewards, dtype=np.float32)
    else:
        return (rewards - mean) / (raw_std + 1e-8)

def grpo_loss(logprobs_current, logprobs_old, advantages, kl_divergences,
              clip_range=0.2, kl_coeff=0.1):
    """
    Computes the GRPO loss, which combines a clipped surrogate objective with a KL penalty.
    """
    # get the ratio
    ratios = mx.exp(logprobs_current - logprobs_old)

    # Surrogate objective
    surr1 = ratios * advantages

    # Manual clamp
    lower = 1.0 - clip_range
    upper = 1.0 + clip_range
    ratios_clamped = mx.minimum(mx.maximum(ratios, lower), upper)
    surr2 = ratios_clamped * advantages

    # Elementwise min => mx.minimum
    surr_min = mx.minimum(surr1, surr2)
    surrogate_loss = -mx.mean(surr_min)

    # KL penalty
    kl_loss = kl_coeff * mx.mean(kl_divergences)

    # return the loss
    return surrogate_loss + kl_loss
