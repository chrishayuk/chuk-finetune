# tests/test_grpo_loss_mlx.py

import pytest
import numpy as np

import mlx.core as mx

from src.train.grpo.mlx.grpo_loss import compute_advantages as mlx_compute_advantages
from src.train.grpo.mlx.grpo_loss import grpo_loss as mlx_grpo_loss

@pytest.mark.parametrize("rewards", [
    [1.0, 2.0, 3.0],
    [-1.0, 0.0, 1.0],
    [10.0, 10.0, 10.0],  # All same
    np.random.randn(5).tolist(),
])
def test_compute_advantages_mlx(rewards):
    ...
    advantages_np = mlx_compute_advantages(rewards)

    assert len(advantages_np) == len(rewards)
    mean_adv = np.mean(advantages_np)
    std_adv = np.std(advantages_np)

    # If all rewards are identical, check that the normalised result is all zeros
    if all(np.isclose(rewards, rewards[0])):
        # They should all be zero
        np.testing.assert_allclose(advantages_np, 0.0, atol=1e-5)
    else:
        # Then we expect near-zero mean, near-one std
        assert abs(mean_adv) < 1e-5, f"Mean of advantages is not near 0, got {mean_adv}"
        assert abs(std_adv - 1.0) < 1e-5, f"Std of advantages is not near 1, got {std_adv}"


def test_grpo_loss_mlx():
    """
    Basic test for MLX grpo_loss to ensure it returns a scalar 
    MLX array (no crashes).
    """
    # Synthetic data
    logprobs_current_np = np.array([0.2, -0.5, 1.0, 0.3], dtype=np.float32)
    logprobs_old_np = np.array([0.0, -0.3, 0.9, 0.4], dtype=np.float32)
    advantages_np = np.array([1.0, -1.0, 0.5, 2.0], dtype=np.float32)
    kl_divs_np = np.array([0.05, 0.1, 0.02, 0.2], dtype=np.float32)

    logprobs_current = mx.array(logprobs_current_np)
    logprobs_old = mx.array(logprobs_old_np)
    advantages = mx.array(advantages_np)
    kl_divs = mx.array(kl_divs_np)

    loss_val = mlx_grpo_loss(
        logprobs_current=logprobs_current,
        logprobs_old=logprobs_old,
        advantages=advantages,
        kl_divergences=kl_divs,
        clip_range=0.2,
        kl_coeff=0.1
    )
    # loss_val should be a scalar MLX array
    assert isinstance(loss_val, mx.array), "Loss must be an MLX array."
    # Convert to Python float
    loss_f = float(loss_val)
    assert loss_f != 0.0, "Loss shouldn't be exactly zero by default."

@pytest.mark.parametrize("clip_range,kl_coeff", [
    (0.1, 0.05),
    (0.2, 0.1),
    (0.3, 0.0),
])
def test_grpo_loss_mlx_varied_params(clip_range, kl_coeff):
    """
    Check MLX GRPO loss with multiple clip_range & kl_coeff for basic sanity.
    """
    current_np = np.array([1.2, 0.7, -0.3], dtype=np.float32)
    old_np = np.array([1.0, 0.6, -0.1], dtype=np.float32)
    adv_np = np.array([1.0, 0.5, -1.0], dtype=np.float32)
    kl_np = np.array([0.02, 0.15, 0.1], dtype=np.float32)

    current = mx.array(current_np)
    old = mx.array(old_np)
    adv = mx.array(adv_np)
    kl = mx.array(kl_np)

    loss_val = mlx_grpo_loss(
        current, old, adv, kl,
        clip_range=clip_range,
        kl_coeff=kl_coeff
    )
    loss_f = float(loss_val)
    assert np.isfinite(loss_f), "Loss must be finite."
    # The main check is no crash and finite scalar
