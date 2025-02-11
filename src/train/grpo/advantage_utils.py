# src/train/grpo/advantage_utils.py
import numpy as np

def compute_advantages(rewards, eps=1e-8):
    # calculate rewards
    rewards = np.array(rewards, dtype=np.float32)

    # If rewards is empty, return empty array
    if rewards.size == 0:
        return rewards  # empty float32 array
    
    # get the mean
    mean = rewards.mean()
    
    # get std deviation
    raw_std = rewards.std()
    
    if raw_std < eps:
        return np.zeros_like(rewards, dtype=np.float32)
    else:
        return (rewards - mean) / (raw_std + eps)

