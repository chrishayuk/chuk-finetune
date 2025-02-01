import pytest
import numpy as np
import mlx.core as mx

from src.train.mlx.grpo_utils import (
    gather_logprobs,
    gather_kl_divergence
)

def test_gather_logprobs_mlx_basic():
    """
    Test gather_logprobs_mlx on small, hand-crafted logits & input IDs.
    """
    # shape: [1, seq_len=3, vocab_size=5]
    # We'll create some randomish values
    logits_np = np.array([[
        [2.0, 1.0, 0.0, -1.0, -2.0],
        [0.0,  0.5, 1.5, -0.5,  2.0],
        [0.1,  0.2, 0.3,  0.4,  0.5]
    ]], dtype=np.float32)
    # Convert to MLX array
    logits = mx.array(logits_np)

    # input_ids: 3 tokens => shape [3]
    input_ids_np = np.array([0, 2, 4], dtype=np.uint32)
    input_ids = mx.array(input_ids_np)

    # call gather_logprobs_mlx
    logprob_sum = gather_logprobs(logits, input_ids)

    # It's an MLX array shape [1], convert to float
    logprob_val = float(logprob_sum)

    # We can check if it runs without error, or compare to a reference.
    assert logprob_sum.shape == (1,)
    # Optionally do a known reference computation:
    # Torch style:
    # 1) log softmax over axis=-1
    # compute manually or use a small Torch snippet if you want
    # For now, let's just ensure it doesn't crash, and returns finite
    assert np.isfinite(logprob_val)

def test_gather_kl_divergence_mlx_basic():
    """
    Test gather_kl_divergence_mlx on small shapes.
    """
    # shape [1, seq_len=2, vocab_size=4]
    current_np = np.array([[
        [0.5, 1.0, -1.5, 2.0],
        [2.0, 0.0,  1.0, -3.0]
    ]], dtype=np.float32)
    ref_np = np.array([[
        [0.0, 0.0,  0.0,  0.0],
        [1.0, 1.0,  1.0,  1.0]
    ]], dtype=np.float32)

    current = mx.array(current_np)
    ref     = mx.array(ref_np)

    # input_ids => 2 tokens
    input_ids = mx.array([3, 1], mx.uint32)

    kl_val_mlx = gather_kl_divergence(current, ref, input_ids)
    assert kl_val_mlx.shape == (1,)
    kl_val = float(kl_val_mlx)
    assert np.isfinite(kl_val), "KL should be finite"

def test_gather_logprobs_mlx_vs_torch():
    """
    (Optional) Compare MLX gather_logprobs to Torch gather_logprobs on random data.
    """
    import torch

    batch = 1
    seq_len = 3
    vocab_size = 5

    # Random logits
    rng = np.random.default_rng(42)
    logits_np = rng.normal(size=(batch, seq_len, vocab_size)).astype(np.float32)

    # random input_ids in [0, vocab_size)
    input_ids_np = rng.integers(low=0, high=vocab_size, size=(seq_len,), dtype=np.uint32)

    # Torch version
    torch_logits = torch.tensor(logits_np)
    torch_logits_lsm = torch_logits.log_softmax(dim=-1)  # [1, seq_len, vocab]
    torch_input_ids = torch.tensor(input_ids_np, dtype=torch.long) # shape [seq_len]
    
    gathered_torch = []
    for t in range(seq_len):
        token_id = torch_input_ids[t]
        # log-prob for that token
        gathered_torch.append(torch_logits_lsm[0, t, token_id])
    sum_torch = torch.stack(gathered_torch).sum()

    # MLX version
    mlx_logits = mx.array(logits_np)
    mlx_input_ids = mx.array(input_ids_np)
    sum_mlx = gather_logprobs(mlx_logits, mlx_input_ids)
    # shape [1]
    sum_mlx_val = float(sum_mlx)

    # Compare
    sum_torch_val = float(sum_torch.item())
    np.testing.assert_allclose(sum_mlx_val, sum_torch_val, atol=1e-5, rtol=1e-5)
