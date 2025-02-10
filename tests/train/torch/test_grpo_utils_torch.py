# tests/train/torch/test_torch_grpo_utils.py

import pytest
import torch
import numpy as np

from train.grpo.torch.grpo_utils import gather_logprobs, gather_kl_divergence


def test_gather_logprobs_small():
    """
    Test gather_logprobs on small, handcrafted logits & token IDs.
    """
    # We'll create shape [1, seq_len=3, vocab_size=5]
    logits_data = torch.tensor([[
        [ 2.0,  1.0,  0.0, -1.0, -2.0],
        [ 0.0,  0.5,  1.5, -0.5,  2.0],
        [ 0.1,  0.2,  0.3,  0.4,  0.5]
    ]], dtype=torch.float32)  # shape [1, 3, 5]

    # input_ids shape [1, 3] (batch_size=1, seq_len=3)
    input_ids_data = torch.tensor([[0, 2, 4]], dtype=torch.long)

    # Call gather_logprobs
    sum_logprob = gather_logprobs(logits_data, input_ids_data)

    # The function should return shape [1]
    assert sum_logprob.shape == (1,), f"Expected shape [1], got {sum_logprob.shape}"
    val = sum_logprob.item()
    assert np.isfinite(val), "Output logprob sum is not finite"


def test_gather_kl_divergence_small():
    """
    Test gather_kl_divergence on small shapes, verifying shape & finiteness.
    """
    # current_logits and ref_logits shape [1, seq_len=2, vocab_size=4]
    current_data = torch.tensor([[
        [ 0.5,  1.0, -1.5,  2.0],
        [ 2.0,  0.0,  1.0, -3.0]
    ]], dtype=torch.float32)
    ref_data = torch.tensor([[
        [ 0.0,  0.0,  0.0,  0.0],
        [ 1.0,  1.0,  1.0,  1.0]
    ]], dtype=torch.float32)

    # input_ids shape [1, 2]
    input_ids_data = torch.tensor([[3, 1]], dtype=torch.long)

    kl_val = gather_kl_divergence(current_data, ref_data, input_ids_data)
    assert kl_val.shape == (1,), f"Expected shape [1], got {kl_val.shape}"
    kl_val_f = kl_val.item()
    assert np.isfinite(kl_val_f), "KL must be finite."


def test_gather_logprobs_random():
    """
    Test gather_logprobs on random data, ensuring no shape or runtime errors.
    """
    batch_size = 1
    seq_len = 4
    vocab_size = 6

    rng = np.random.default_rng(12345)
    # create random logits => shape [1, 4, 6]
    logits_np = rng.normal(loc=0.0, scale=1.0, size=(batch_size, seq_len, vocab_size)).astype(np.float32)
    logits_t = torch.tensor(logits_np)

    # create random input_ids => shape [1, 4]
    input_ids_np = rng.integers(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    input_ids_t = torch.tensor(input_ids_np, dtype=torch.long)

    sum_lp = gather_logprobs(logits_t, input_ids_t)
    assert sum_lp.shape == (1,)
    val = sum_lp.item()
    assert np.isfinite(val), "sum of log-probs is not finite"


def test_gather_kl_divergence_random():
    """
    Test gather_kl_divergence on random data, ensuring no shape or runtime errors.
    """
    rng = np.random.default_rng(999)
    batch_size = 1
    seq_len = 3
    vocab_size = 5

    current_np = rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32)
    ref_np     = rng.normal(size=(batch_size, seq_len, vocab_size)).astype(np.float32)

    current_t = torch.tensor(current_np)
    ref_t     = torch.tensor(ref_np)

    # random input_ids => shape [1, 3]
    input_ids_np = rng.integers(low=0, high=vocab_size, size=(batch_size, seq_len), dtype=np.int64)
    input_ids_t = torch.tensor(input_ids_np, dtype=torch.long)

    kl_val = gather_kl_divergence(current_t, ref_t, input_ids_t)
    assert kl_val.shape == (1,)
    assert np.isfinite(kl_val.item()), "KL not finite."
