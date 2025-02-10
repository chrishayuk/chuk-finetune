# tests/test_grpo_loss_torch.py

import pytest
import numpy as np
import torch

from train.grpo.advantage_utils import compute_advantages
from train.grpo.torch.grpo_loss import grpo_loss as torch_grpo_loss

@pytest.mark.parametrize("rewards", [
    [1.0, 2.0, 3.0],
    [-1.0, 0.0, 1.0],
    [10.0, 10.0, 10.0],  # All same
    np.random.randn(5).tolist(),
])
def test_compute_advantages(rewards):
    advantages = compute_advantages(rewards)
    assert len(advantages) == len(rewards)

    # If rewards are all identical, the normalised result is all zeros => std=0
    if all(np.isclose(rewards, rewards[0])):
        # Check that advantages are all zero
        np.testing.assert_allclose(advantages, 0.0, atol=1e-5)
    else:
        # Otherwise we expect near-zero mean, near-one std
        np.testing.assert_allclose(np.mean(advantages), 0.0, atol=1e-5)
        np.testing.assert_allclose(np.std(advantages), 1.0, atol=1e-5)
        
def test_grpo_loss_torch():
    """
    Basic test for Torch grpo_loss to ensure no crashes & plausible scalar output.
    """
    # Synthetic data
    logprobs_current = torch.tensor([0.2, -0.5, 1.0, 0.3])
    logprobs_old = torch.tensor([0.0, -0.3, 0.9, 0.4])
    advantages = torch.tensor([1.0, -1.0, 0.5, 2.0])
    kl_divs = torch.tensor([0.05, 0.1, 0.02, 0.2])
    clip_range = 0.2
    kl_coeff = 0.1

    loss_val = torch_grpo_loss(
        logprobs_current=logprobs_current,
        logprobs_old=logprobs_old,
        advantages=advantages,
        kl_divergences=kl_divs,
        clip_range=clip_range,
        kl_coeff=kl_coeff
    )
    assert loss_val.dim() == 0, "Loss should be a scalar tensor."
    assert loss_val.item() != 0.0, "Loss shouldn't be exactly zero by default."

@pytest.mark.parametrize("clip_range,kl_coeff", [
    (0.1, 0.05),
    (0.2, 0.1),
    (0.3, 0.0),
])
def test_grpo_loss_torch_varied_params(clip_range, kl_coeff):
    """
    Check Torch GRPO loss with varied clip_range & kl_coeff for basic sanity.
    """
    current = torch.tensor([1.2, 0.7, -0.3])
    old = torch.tensor([1.0, 0.6, -0.1])
    adv = torch.tensor([1.0, 0.5, -1.0])
    kl = torch.tensor([0.02, 0.15, 0.1])

    loss_val = torch_grpo_loss(current, old, adv, kl,
                               clip_range=clip_range, kl_coeff=kl_coeff)
    assert loss_val.dim() == 0
    assert torch.isfinite(loss_val), "Loss must be finite."
