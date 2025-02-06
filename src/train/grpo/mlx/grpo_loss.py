# src/train/grpo/mlx/grpo_loss.py
import logging
import numpy as np
import mlx.core as mx

logger = logging.getLogger(__name__)

def gather_logprobs(logits, input_ids):
    """
    MLX version of gather_logprobs:
      1) Manually compute log softmax => logprobs = logits - logsumexp(logits, axis=-1)
      2) For each token i in input_ids, sum logprobs[0, t, token_id].
      3) Return shape [1].
    ---------------------------------------------------------------------------
    :param logits: shape [1, seq_len, vocab_size], unnormalized token logits.
    :param input_ids: Python list of token IDs, length seq_len.
    """
    # 1) manual log softmax => shape [1, seq_len, vocab_size]
    logsumexp_vals = mx.logsumexp(logits, axis=-1, keepdims=True)
    logprobs = logits - logsumexp_vals

    # 2) sum up log-probs
    seq_len = len(input_ids)
    total_logprob = mx.array(0.0, mx.float32)

    for t in range(seq_len):
        token_id = input_ids[t]
        # accumulate logprobs[0, t, token_id]
        total_logprob += logprobs[0, t, token_id]

    # Return as shape [1]
    return total_logprob[None]

def gather_kl_divergence(current_logits, ref_logits, input_ids):
    """
    Computes the average KL divergence between current model and ref_model
    over the given token sequence.

    :param current_logits: shape [1, seq_len, vocab_size].
    :param ref_logits: same shape as current_logits.
    :param input_ids: Python list of token IDs, length seq_len.
    :return: shape [1] containing average KL over tokens in input_ids.
    """
    current_lse = mx.logsumexp(current_logits, axis=-1, keepdims=True)
    current_logprobs = current_logits - current_lse

    ref_lse = mx.logsumexp(ref_logits, axis=-1, keepdims=True)
    ref_logprobs = ref_logits - ref_lse

    seq_len = len(input_ids)
    total_kl = mx.array(0.0, mx.float32)

    for t in range(seq_len):
        token_id = input_ids[t]
        log_diff = current_logprobs[0, t, token_id] - ref_logprobs[0, t, token_id]
        total_kl += log_diff

    # Normalize by seq_len (optional, based on your approach)
    return (total_kl / seq_len)[None]

def compute_advantages(rewards):
    """
    Basic reward normalization to produce advantages:
      advantages = (rewards - mean) / (std + 1e-8)
    """
    arr = np.array(rewards, dtype=np.float32)
    arr = (arr - arr.mean()) / (arr.std() + 1e-8)
    return arr

def grpo_loss(
    logprobs_current,
    logprobs_old,
    advantages,
    kl_divergences,
    clip_range=0.2,
    kl_coeff=0.1
):
    """
    GRPO/PPO-style loss with ratio clipping and KL penalty.
    Assumes logprobs_* are negative log-likelihood values; 
    hence ratio = exp(logprobs_old - logprobs_current).
    """
    # ratio
    ratios = mx.exp(logprobs_old - logprobs_current)

    # Surrogate objective
    surr1 = ratios * advantages

    # Clamping
    lower = 1.0 - clip_range
    upper = 1.0 + clip_range
    ratios_clamped = mx.minimum(mx.maximum(ratios, lower), upper)
    surr2 = ratios_clamped * advantages

    # Elementwise minimum
    surr_min = mx.minimum(surr1, surr2)
    surrogate_loss = -mx.mean(surr_min)

    # KL penalty
    kl_loss = kl_coeff * mx.mean(kl_divergences)

    # Final loss
    return surrogate_loss + kl_loss

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
      1) Encodes each response, computes new logprobs vs. current model.
      2) Computes KL divergence vs. ref model.
      3) Normalizes 'rewards' => advantages, calls 'grpo_loss'.
    """
    # 1) Compute advantages
    advantages_arr = compute_advantages(rewards)

    # 2) For each response, gather new logprobs and KL
    current_list = []
    kl_list = []

    for resp in responses:
        tokens = tokenizer.encode(resp)
        if not tokens:
            tokens = [tokenizer.eos_token_id]

        out_current = model(mx.array(tokens, mx.uint32)[None])
        sum_current = gather_logprobs(out_current, tokens)

        out_ref = ref_model(mx.array(tokens, mx.uint32)[None])
        kl_val = gather_kl_divergence(out_current, out_ref, tokens)

        current_list.append(sum_current)
        kl_list.append(kl_val)

    logprobs_current_sums = mx.concat(current_list, axis=0)
    kl_sums = mx.concat(kl_list, axis=0)

    old_sums = mx.array(old_logprobs)
    advantages_m = mx.array(advantages_arr)

    # 3) Call grpo_loss
    loss_val = grpo_loss(
        logprobs_current=logprobs_current_sums,
        logprobs_old=old_sums,
        advantages=advantages_m,
        kl_divergences=kl_sums,
        clip_range=0.2,  # or tune as desired
        kl_coeff=kl_coeff
    )

    return loss_val

def single_question_loss(
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
    Simple wrapper that delegates to 'compute_grpo_loss'.
    """
    return compute_grpo_loss(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        item=item,
        responses=responses,
        old_logprobs=old_logprobs,
        rewards=rewards,
        kl_coeff=kl_coeff,
        verbose=verbose
    )