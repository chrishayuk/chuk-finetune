# src/train/grpo/advantage_utils.py
import numpy as np

def compute_advantages(rewards, eps=1e-8):
    """
    Normalizes the input rewards by subtracting the mean and dividing by the 
    standard deviation. Accepts a Python list or a NumPy array.
    Returns a NumPy array.
    """
    rewards = np.array(rewards, dtype=np.float32)

    mean = rewards.mean()
    raw_std = rewards.std()
    
    if raw_std < eps:
        return np.zeros_like(rewards, dtype=np.float32)
    else:
        return (rewards - mean) / (raw_std + eps)
