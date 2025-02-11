# tests/test_advantage_utils.py

import numpy as np
import pytest

from train.grpo.advantage_utils import compute_advantages

def test_compute_advantages_basic():
    """
    Test with a typical list of rewards. 
    Check that output is zero-mean and unit-std.
    """
    rewards = [1, 2, 3, 4, 5]
    advantages = compute_advantages(rewards)

    # Check type & shape
    assert isinstance(advantages, np.ndarray), "Output should be a NumPy array."
    assert advantages.shape == (5,), "Output shape should match input length."

    # Mean should be close to 0
    assert abs(advantages.mean()) < 1e-6, "Expected near-zero mean after normalization."
    # Std should be close to 1
    assert abs(advantages.std() - 1) < 1e-6, "Expected near-one std after normalization."

def test_compute_advantages_with_np_array():
    """
    Test with direct NumPy array input instead of a list,
    ensuring the function handles both seamlessly.
    """
    rewards_np = np.array([10, 20, 30, 40, 50], dtype=np.float32)
    advantages = compute_advantages(rewards_np)

    assert isinstance(advantages, np.ndarray)
    assert advantages.shape == (5,)

    # Similarly check near-zero mean, near-one std
    assert abs(advantages.mean()) < 1e-6
    assert abs(advantages.std() - 1) < 1e-6

def test_compute_advantages_single_value():
    """
    Single value => std = 0 => compute_advantages returns zeros.
    """
    rewards = [10]
    advantages = compute_advantages(rewards)

    assert advantages.shape == (1,)
    # All zeros
    assert np.allclose(advantages, 0), "Expected zero advantage if std=0."

def test_compute_advantages_all_same():
    """
    Multiple identical values => std=0 => zero advantage array.
    """
    rewards = [5, 5, 5, 5]
    advantages = compute_advantages(rewards)

    assert advantages.shape == (4,)
    # All zeros
    assert np.allclose(advantages, 0), "Expected zero advantage if all values are the same."

def test_compute_advantages_empty():
    """
    Edge case: empty input => returns empty array.
    """
    advantages = compute_advantages([])
    assert advantages.size == 0, "Expected empty output for empty input."
    assert advantages.dtype == np.float32, "Expected float32 dtype even if empty."

def test_compute_advantages_negative_and_positive():
    """
    Mix of negative, zero, and positive => check shape & near-zero mean / near-one std.
    """
    rewards = [-1, 0, 1]
    advantages = compute_advantages(rewards)

    assert advantages.shape == (3,)
    assert abs(advantages.mean()) < 1e-6
    assert abs(advantages.std() - 1) < 1e-6
