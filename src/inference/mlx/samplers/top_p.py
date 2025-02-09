# src/inference/mlx/samplers/top_p.py

import mlx.core as mx
from inference.prompt_removal import remove_prompt_prefix
from inference.stop_utils import check_stop_sequences, prepare_stop_sequences

def top_p_generate(
    model,
    tokenizer,
    prompt,
    max_tokens=200,
    top_p=0.9,
    temperature=1.0,
    stop_sequences=None
):
    """
    Token-by-token top-p (nucleus) sampling (ASCENDING order) in MLX.

    We fix potential zero-distribution issues by forcing the last token 
    (highest probability in ascending order) to be kept if cumsum never 
    exceeds (1 - top_p).
    """
    if stop_sequences is None:
        stop_sequences = []
    stop_sequences = prepare_stop_sequences(stop_sequences)

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.eos_token_id]

    for _ in range(max_tokens):
        # 1) Forward pass -> pass entire token list (no KV-cache)
        logits = model(mx.array(tokens, mx.uint32)[None])

        # 2) Logits for the last token in the sequence
        last_logits = logits[:, -1, :]

        # 3) Apply temperature
        scaled_logits = last_logits / temperature

        # 4) Convert logits -> probabilities
        probs = mx.softmax(scaled_logits, axis=-1)  # shape [1, vocab_size]

        # 5) Sort in ascending order
        sorted_indices = mx.argsort(probs, axis=-1)  # ascending
        sorted_probs   = mx.take_along_axis(probs, sorted_indices, axis=-1)

        # 6) Cumulative sum in ascending order
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

        # 7) In ascending order, the top-p portion is the tail where 
        #    cumsum > (1.0 - top_p). We want to keep that tail.
        threshold_mask = cumulative_probs > (1.0 - top_p)

        # 8) If threshold_mask is all False, it means we never exceeded (1 - top_p).
        #    In that edge case, we force ourselves to keep the last token 
        #    (the highest probability in ascending order).
        if (threshold_mask.sum(axis=-1).item() == 0):
            # Force last token to be included
            threshold_mask[..., -1] = True

        # 9) Zero out everything *before* we exceed that threshold
        truncated_probs = mx.where(threshold_mask, sorted_probs, mx.array(0.0, probs.dtype))

        # 10) Re-normalize to sum to 1
        sum_trunc = truncated_probs.sum(axis=-1, keepdims=True) + 1e-12
        truncated_probs = truncated_probs / sum_trunc

        # 11) Sample from truncated & re-normalized distribution
        chosen_index = mx.random.categorical(mx.log(truncated_probs + 1e-12), axis=-1).item()

        # 12) Map sampled index back to actual token ID
        chosen_index_arr = mx.array(chosen_index, mx.uint32).reshape((1, 1))
        next_token = mx.take_along_axis(sorted_indices, chosen_index_arr, axis=-1).item()

        tokens.append(next_token)

        # 13) Early stopping: check EOS
        if next_token == tokenizer.eos_token_id:
            break

        # 14) Check custom stop sequences
        current_text = tokenizer.decode(tokens)
        maybe_truncated = check_stop_sequences(current_text, stop_sequences)
        if maybe_truncated is not None:
            return maybe_truncated

    # 15) Final decode + remove prompt prefix
    raw_output = tokenizer.decode(tokens)
    return remove_prompt_prefix(raw_output, prompt)
