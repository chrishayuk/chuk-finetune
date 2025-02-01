# src/train/torch/grpo_loss.py
import torch
import numpy as np

def compute_advantages(rewards):
    """
    Normalises the input rewards by subtracting the mean and dividing by the standard deviation.
    """
    # get the rewards
    rewards = np.array(rewards, dtype=np.float32)

    # get the mean
    mean = rewards.mean()

    # get standard deviation
    std = rewards.std() + 1e-8  # Avoid division by zero

    # subtract the mean from the rewards and divide by standard deviation
    if std < 1e-8:
        # All values basically the same, so normalised = all zeros
        return np.zeros_like(rewards, dtype=np.float32)
    else:
        return (rewards - mean) / (std + 1e-8)


def grpo_loss(logprobs_current, logprobs_old, advantages, kl_divergences, clip_range=0.2, kl_coeff=0.1):
    """
    Computes the GRPO loss, which combines a clipped surrogate objective with a KL penalty.
    """
    # get the ratio
    ratios = torch.exp(logprobs_current - logprobs_old)
    
    # Clipped objective
    surr1 = ratios * advantages
    surr2 = torch.clamp(ratios, 1 - clip_range, 1 + clip_range) * advantages
    surrogate_loss = -torch.min(surr1, surr2).mean()
    
    # KL penalty
    kl_loss = kl_coeff * kl_divergences.mean()

    # return the loss
    return surrogate_loss + kl_loss
