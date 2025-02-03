# src/train/mlx/grpo_utils.py
import mlx.core as mx
import numpy as np

def gather_logprobs(logits, input_ids):
    """
    MLX version of gather_logprobs:
    1) manual log softmax => logprobs = logits - logsumexp(logits, axis=-1)
    2) for each token i in input_ids, sum logprobs[0,t,token].
    3) return shape [1].
    """
    # logits: shape [1, seq_len, vocab_size]
    # input_ids: a Python list of length seq_len.

    # 1) manual log softmax => shape [1,seq_len,vocab_size]
    logsumexp_vals = mx.logsumexp(logits, axis=-1, keepdims=True)
    logprobs = logits - logsumexp_vals

    # 2) sum up log-probs
    seq_len = len(input_ids)  # fix: input_ids is a list
    total_logprob = mx.array(0.0, mx.float32)

    for t in range(seq_len):
        token_id = input_ids[t]
        # accumulate logprobs[0, t, token_id]
        total_logprob += logprobs[0, t, token_id]

    # shape [1]
    return total_logprob[None]

def gather_kl_divergence(current_logits, ref_logits, input_ids):
    """
    MLX version of KL. Same pattern: input_ids is a list, so do len(input_ids).
    """
    current_logsumexp = mx.logsumexp(current_logits, axis=-1, keepdims=True)
    current_logprobs  = current_logits - current_logsumexp

    ref_logsumexp = mx.logsumexp(ref_logits, axis=-1, keepdims=True)
    ref_logprobs   = ref_logits - ref_logsumexp

    seq_len = len(input_ids)  # fix
    total_kl = mx.array(0.0, mx.float32)

    for t in range(seq_len):
        token_id = input_ids[t]
        p_x = mx.exp(current_logprobs[0, t, token_id])
        log_ratio = current_logprobs[0, t, token_id] - ref_logprobs[0, t, token_id]
        total_kl += p_x * log_ratio

    return total_kl[None]
