# src/train/grpo/mlx/grpo_loss.py
import numpy as np
import mlx.core as mx

def compute_advantages(rewards):
    """
    Basic advantage function that just returns the raw rewards.
    If your reward function yields variation among responses,
    you'll avoid zero advantages.
    
    If all responses in a batch receive the same reward,
    advantages will be identical (and can be zero if you do mean subtraction).
    """
    return np.array(rewards, dtype=np.float32)

def grpo_loss(logprobs_current, logprobs_old, advantages, kl_divergences,
              clip_range=0.2, kl_coeff=0.1):
    # ratio
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

    # Final loss is policy surrogate plus KL penalty
    return surrogate_loss + kl_loss
