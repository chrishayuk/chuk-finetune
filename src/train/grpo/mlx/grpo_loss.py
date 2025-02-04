# src/train/grpo/mlx/grpo_loss.py

import numpy as np
import mlx.core as mx

def compute_advantages(rewards):
    arr = np.array(rewards, dtype=np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return arr

def grpo_loss(logprobs_current, logprobs_old, advantages, kl_divergences,
              clip_range=0.2, kl_coeff=0.1):
    """
    GRPO/PPO-style loss with ratio clipping and KL penalty.
    Assumes logprobs_* are negative log-likelihood values; 
    hence the ratio is exp(logprobs_old - logprobs_current).
    """

    # Fix: reversed the ratio because gather_logprobs returns negative log-likelihood.
    ratios = mx.exp(logprobs_old - logprobs_current)

    # Surrogate objective
    surr1 = ratios * advantages

    # Clamping
    lower = 1.0 - clip_range
    upper = 1.0 + clip_range
    ratios_clamped = mx.minimum(mx.maximum(ratios, lower), upper)
    surr2 = ratios_clamped * advantages

    # Elementwise min => mx.minimum
    surr_min = mx.minimum(surr1, surr2)
    surrogate_loss = -mx.mean(surr_min)

    # KL penalty
    kl_loss = kl_coeff * mx.mean(kl_divergences)

    # Final loss
    return surrogate_loss + kl_loss
