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
    Token-by-token top-p (nucleus) sampling.
    """
    if stop_sequences is None:
        stop_sequences = []
    stop_sequences = prepare_stop_sequences(stop_sequences)

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.eos_token_id]

    for _ in range(max_tokens):
        logits = model(mx.array(tokens, mx.uint32)[None])
        last_logits = logits[:, -1, :]
        scaled_logits = last_logits * (1.0 / temperature)

        # Convert logits -> probabilities
        probs = mx.softmax(scaled_logits, axis=-1)

        # Sort tokens by ascending probability
        sorted_indices = mx.argsort(probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

        # Cumulative sum
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

        # Nucleus cutoff
        threshold_mask = cumulative_probs > (1.0 - top_p)
        safe_probs = mx.where(threshold_mask, sorted_probs, mx.array(0.0, probs.dtype))

        # Sample
        chosen_index = mx.random.categorical(mx.log(safe_probs + 1e-12), axis=-1).item()
        chosen_index_arr = mx.array(chosen_index, mx.uint32).reshape((1, 1))
        next_token = mx.take_along_axis(sorted_indices, chosen_index_arr, axis=-1).item()

        tokens.append(next_token)

        # Check EOS
        if next_token == tokenizer.eos_token_id:
            break

        # Check stops
        current_text = tokenizer.decode(tokens)
        maybe_truncated = check_stop_sequences(current_text, stop_sequences)
        if maybe_truncated is not None:
            return maybe_truncated

    # Final decode + remove prefix
    raw_output = tokenizer.decode(tokens)
    final_output = remove_prompt_prefix(raw_output, prompt)
    return final_output
