# src/train/grpo/mlx/grpo_loss.py

import logging
import numpy as np
import mlx.core as mx

# Import the shared advantage computation function
from train.grpo.advantage_utils import compute_advantages

logger = logging.getLogger(__name__)

def gather_logprobs(logits, input_ids):
    """
    Compute the sum of log-probabilities for the given 'input_ids'.
    1) Manually compute log softmax => logprobs = logits - logsumexp(logits, axis=-1)
    2) For each token in 'input_ids', accumulate logprobs[0, t, token_id].
    3) Return shape [1] containing the total log-prob.
    """
    # 1) log-softmax => shape [1, seq_len, vocab_size]
    logsumexp_vals = mx.logsumexp(logits, axis=-1, keepdims=True)
    logprobs = logits - logsumexp_vals  # element-wise

    # 2) sum up log-probs for each token
    seq_len = len(input_ids)
    total_logprob = mx.array(0.0, mx.float32)
    for t in range(seq_len):
        token_id = input_ids[t]
        total_logprob += logprobs[0, t, token_id]

    # Return shape [1]
    return total_logprob[None]


def gather_kl_divergence(current_logits, ref_logits, input_ids):
    """
    Computes (optionally average) KL divergence between 'current_logits' and 'ref_logits'
    over the sequence in 'input_ids'.

    1) Convert both logits to log-probs via (logits - logsumexp(logits)).
    2) For each token, accumulate [logprob_current - logprob_ref].
    3) Optionally divide by seq_len to get an average.
    4) Return shape [1].
    """
    # Convert logits to log-probs
    current_lse = mx.logsumexp(current_logits, axis=-1, keepdims=True)
    ref_lse = mx.logsumexp(ref_logits, axis=-1, keepdims=True)

    current_logprobs = current_logits - current_lse
    ref_logprobs = ref_logits - ref_lse

    seq_len = len(input_ids)
    total_kl = mx.array(0.0, mx.float32)

    for t in range(seq_len):
        token_id = input_ids[t]
        # KL contribution for token_id at time t
        log_diff = current_logprobs[0, t, token_id] - ref_logprobs[0, t, token_id]
        total_kl += log_diff

    # Here we do average KL per token
    return (total_kl / seq_len)[None]


def grpo_loss(
    logprobs_current,
    logprobs_old,
    advantages,
    kl_divergences,
    clip_range=0.2,
    kl_coeff=0.1,
    reduction="mean"
):
    """
    GRPO/PPO-style objective with ratio clipping + KL penalty.

    Objective:
        L = - E[ min(r * A, clip(r, 1-ε, 1+ε) * A ) ] + kl_coeff * KL

    Where:
      - r = exp(logπ_new - logπ_old)
      - A = advantage
      - KL = per-sample KL divergence with the reference policy

    :param logprobs_current: shape [N], log π_new for each sample
    :param logprobs_old: shape [N], log π_old for each sample
    :param advantages: shape [N], advantage values
    :param kl_divergences: shape [N], KL per sample
    :param clip_range: float, PPO clipping parameter
    :param kl_coeff: float, weighting for KL divergence
    :param reduction: 'mean', 'sum', or 'none'
    :return: scalar (if 'mean' or 'sum') or per-sample if 'none'
    """
    # ratio = exp(logπ_new - logπ_old)
    ratios = mx.exp(logprobs_current - logprobs_old)  # shape [N]

    # Surrogate
    surr1 = ratios * advantages
    # clamp(ratios, 1-clip, 1+clip)
    ratios_clamped = mx.minimum(mx.maximum(ratios, 1.0 - clip_range), 1.0 + clip_range)
    surr2 = ratios_clamped * advantages

    # Negative sign => we want to maximize => we minimize the negative
    clipped_surrogate = -mx.minimum(surr1, surr2)

    # KL penalty term
    kl_term = kl_coeff * kl_divergences

    # Combined loss per sample
    loss = clipped_surrogate + kl_term  # shape [N]

    # Reduction
    if reduction == "mean":
        return mx.mean(loss)
    elif reduction == "sum":
        return mx.sum(loss)
    else:
        # 'none' => return the vector of per-sample losses
        return loss


def compute_grpo_loss(
    model,
    ref_model,
    tokenizer,
    item,
    responses,
    old_logprobs,
    rewards,
    kl_coeff=0.1,
    verbose=False
):
    """
    High-level function that:
      1) Normalizes 'rewards' => advantages via shared NumPy-based utility.
      2) For each response, compute new logprobs & KL vs. reference model.
      3) Computes the GRPO loss by comparing old vs. current logprobs,
         factoring in the KL penalty.

    :param model: MLX model (current policy).
    :param ref_model: MLX model (reference/old policy).
    :param tokenizer: tokenizer with an encode method that returns list of token IDs.
    :param item: original question/data item (not used here, but included for clarity).
    :param responses: list of text responses (strings).
    :param old_logprobs: list of floats, old log-probs for each response.
    :param rewards: list of floats, reward signal for each response.
    :param kl_coeff: weighting factor for the KL penalty.
    :param verbose: bool, if True, may print debugging info.
    :return: Scalar GRPO loss (MLX float).
    """
    # 1) Compute advantages (NumPy-based) from the shared utility
    advantages_arr = compute_advantages(rewards)  # returns a NumPy array

    # 2) Gather new logprobs + KL for each response
    current_logprob_list = []
    kl_list = []

    for resp in responses:
        tokens = tokenizer.encode(resp)
        if not tokens:
            # Fallback to an EOS token if blank
            tokens = [tokenizer.eos_token_id]

        out_current = model(mx.array(tokens, mx.uint32)[None])   # shape [1, seq_len, vocab]
        sum_current = gather_logprobs(out_current, tokens)       # shape [1]

        out_ref = ref_model(mx.array(tokens, mx.uint32)[None])   # shape [1, seq_len, vocab]
        kl_val = gather_kl_divergence(out_current, out_ref, tokens)  # shape [1]

        current_logprob_list.append(sum_current)
        kl_list.append(kl_val)

    # Concatenate => shape [N] (where N = # of responses)
    logprobs_current_sums = mx.concat(current_logprob_list, axis=0)
    kl_sums = mx.concat(kl_list, axis=0)

    # Convert old_logprobs + advantages to MLX arrays
    old_sums = mx.array(old_logprobs, mx.float32)  
    advantages_m = mx.array(advantages_arr, mx.float32)

    # 3) Compute final GRPO loss (mean over samples)
    loss_val = grpo_loss(
        logprobs_current=logprobs_current_sums,
        logprobs_old=old_sums,
        advantages=advantages_m,
        kl_divergences=kl_sums,
        clip_range=0.2,
        kl_coeff=kl_coeff,
        reduction="mean"
    )

    return loss_val