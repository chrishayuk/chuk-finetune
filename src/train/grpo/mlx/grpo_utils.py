# src/train/grpo/mlx/grpo_utils.py

import mlx.core as mx

def gather_logprobs(logits, input_ids):
    """
    Batched sum of log-probabilities.
    :param logits: shape [B, seq_len, vocab_size]
    :param input_ids: shape [B, seq_len]
    :return: shape [B], sum of log-probs for each sequence in the batch.
    """
    # 1) log-softmax => shape [B, seq_len, vocab_size]
    logsumexp_vals = mx.logsumexp(logits, axis=-1, keepdims=True)
    logprobs = logits - logsumexp_vals

    B, seq_len = input_ids.shape
    sums = mx.zeros([B], mx.float32)

    for b in range(B):
        for t in range(seq_len):
            token_id = input_ids[b, t]
            sums[b] += logprobs[b, t, token_id]
    return sums


def gather_kl_divergence(current_logits, ref_logits, input_ids):
    """
    Batched KL => sum_{t} [ p_new(x_t) * (log p_new(x_t) - log p_ref(x_t)) ].

    :param current_logits: [B, T, V]
    :param ref_logits: [B, T, V]
    :param input_ids: [B, T]
    :return: shape [B], summation of KL across each token in each sequence.
    """
    # -- 1) log-softmax for both
    curr_lse = mx.logsumexp(current_logits, axis=-1, keepdims=True)  # [B, T, 1]
    ref_lse = mx.logsumexp(ref_logits, axis=-1, keepdims=True)
    curr_logprobs = current_logits - curr_lse  # [B, T, V]
    ref_logprobs = ref_logits - ref_lse        # [B, T, V]

    B, seq_len = input_ids.shape
    kl_vals = mx.zeros([B], mx.float32)

    # -- 2) For each token, gather log p_new(x_t) and log p_ref(x_t)
    #       p_new(x_t) = exp(log p_new(x_t))
    for b in range(B):
        for t in range(seq_len):
            token_id = input_ids[b, t]
            log_p_new = curr_logprobs[b, t, token_id]
            log_p_ref = ref_logprobs[b, t, token_id]
            p_new = mx.exp(log_p_new)
            kl_vals[b] += p_new * (log_p_new - log_p_ref)

    return kl_vals