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
    Token-by-token top-p (nucleus) sampling in MLX (ASCENDING order).

    Safeguards against NaNs/Inf or zero distributions by:
      1) Clamping logits before softmax
      2) Forcing at least one token to remain if threshold_mask is all false
      3) If sum of truncated distribution < 1e-12, fallback to the last token
    """
    if stop_sequences is None:
        stop_sequences = []
    stop_sequences = prepare_stop_sequences(stop_sequences)

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.eos_token_id]

    for _ in range(max_tokens):
        # 1) Forward pass (no KV-cache); pass entire token list
        logits = model(mx.array(tokens, mx.uint32)[None])  # shape [1, seq_len, vocab_size]

        # 2) Take logits for the last token
        last_logits = logits[:, -1, :]  # shape [1, vocab_size]

        # 3) Apply temperature
        scaled_logits = last_logits / temperature

        # --- NEW: clamp logits to avoid overflow in softmax ---
        scaled_logits = mx.clip(scaled_logits, mx.array(-100.0), mx.array(100.0))

        # 4) Convert logits -> probabilities
        probs = mx.softmax(scaled_logits, axis=-1)  # shape [1, vocab_size]

        # 5) Sort in ascending order
        sorted_indices = mx.argsort(probs, axis=-1)  # ascending
        sorted_probs   = mx.take_along_axis(probs, sorted_indices, axis=-1)

        # 6) Cumulative sum in ascending order
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

        # 7) The top-p tail is where cumsum > (1.0 - top_p).
        threshold_mask = cumulative_probs > (1.0 - top_p)

        # 8) If threshold_mask is all False, 
        #    forcibly keep the last token (highest prob in ascending order).
        sum_mask = threshold_mask.sum(axis=-1)
        if float(sum_mask.asnumpy()[0]) == 0:
            # Force last token to be included
            threshold_mask[..., -1] = True

        # 9) Zero out everything *before* that threshold
        truncated_probs = mx.where(threshold_mask, sorted_probs, mx.array(0.0, probs.dtype))

        # 10) Re-normalize
        sum_trunc = truncated_probs.sum(axis=-1, keepdims=True)
        # Convert to Python float for comparison
        sum_value = float(sum_trunc.asnumpy()[0, 0])
        if sum_value < 1e-12:
            # Fallback: pick the last token in ascending order
            # (the highest-prob token). We skip sampling.
            chosen_index = truncated_probs.shape[-1] - 1
        else:
            truncated_probs = truncated_probs / (sum_trunc + 1e-12)

            # 11) Sample from truncated & re-normalized distribution
            chosen_index = mx.random.categorical(mx.log(truncated_probs + 1e-12), axis=-1).item()

        # 12) Map the chosen index back to the actual token ID
        chosen_index_arr = mx.array(chosen_index, mx.uint32).reshape((1, 1))
        next_token = mx.take_along_axis(sorted_indices, chosen_index_arr, axis=-1).item()

        tokens.append(next_token)

        # 13) Check EOS
        if next_token == tokenizer.eos_token_id:
            break

        # 14) Check custom stop sequences
        current_text = tokenizer.decode(tokens)
        maybe_truncated = check_stop_sequences(current_text, stop_sequences)
        if maybe_truncated is not None:
            return maybe_truncated

    # 15) Final decode + remove prompt prefix
    raw_output = tokenizer.decode(tokens)
    final_output = remove_prompt_prefix(raw_output, prompt)
    return final_output
