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
    Token-by-token top-p (nucleus) sampling (ASCENDING order).
    """
    if stop_sequences is None:
        stop_sequences = []
    stop_sequences = prepare_stop_sequences(stop_sequences)

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.eos_token_id]

    for _ in range(max_tokens):
        # Forward pass - pass entire token list (no KV-cache)
        logits = model(mx.array(tokens, mx.uint32)[None])

        # Take the logits for the *last* token
        last_logits = logits[:, -1, :]
        
        # Apply temperature
        scaled_logits = last_logits / temperature

        # Convert logits -> probabilities
        probs = mx.softmax(scaled_logits, axis=-1)  # shape: [1, vocab_size]

        # 1) Sort in ascending order
        sorted_indices = mx.argsort(probs, axis=-1)  # no descending=True needed
        sorted_probs   = mx.take_along_axis(probs, sorted_indices, axis=-1)

        # 2) Cumulative sum in ascending order
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

        # 3) In ascending order, the top-p portion is where cumsum > (1 - top_p)
        #    (We keep only the "tail" of the distribution.)
        threshold_mask = cumulative_probs > (1.0 - top_p)

        # 4) Zero out everything *before* we exceed that threshold
        truncated_probs = mx.where(threshold_mask, sorted_probs, mx.array(0.0, probs.dtype))

        # 5) Re-normalize so truncated distribution sums to 1
        sum_trunc = truncated_probs.sum(axis=-1, keepdims=True) + 1e-12
        truncated_probs = truncated_probs / sum_trunc

        # 6) Sample from truncated & re-normalized distribution
        chosen_index = mx.random.categorical(mx.log(truncated_probs + 1e-12), axis=-1).item()

        # Map sampled index back to actual token ID
        chosen_index_arr = mx.array(chosen_index, mx.uint32).reshape((1, 1))
        next_token = mx.take_along_axis(sorted_indices, chosen_index_arr, axis=-1).item()

        # Append new token
        tokens.append(next_token)

        # Early stopping: check EOS
        if next_token == tokenizer.eos_token_id:
            break

        # Stop sequences
        current_text = tokenizer.decode(tokens)
        maybe_truncated = check_stop_sequences(current_text, stop_sequences)
        if maybe_truncated is not None:
            return maybe_truncated

    # Final decode + remove prompt prefix
    raw_output = tokenizer.decode(tokens)
    final_output = remove_prompt_prefix(raw_output, prompt)
    return final_output


