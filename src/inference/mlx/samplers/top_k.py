# src/inference/mlx/samplers/top_k.py
import mlx.core as mx
from inference.prompt_removal import remove_prompt_prefix
from inference.stop_utils import check_stop_sequences, prepare_stop_sequences

def top_k_generate(
    model,
    tokenizer,
    prompt,
    max_tokens=200,
    top_k=5,
    temperature=1.0,
    stop_sequences=None
):
    """
    Token-by-token top-k sampling.
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

        # Top-k filtering
        kth_val = mx.topk(scaled_logits, k=top_k, axis=-1,
                          largest=True, sorted=False)["values"][:, -1]
        mask = scaled_logits < kth_val
        scaled_logits = mx.where(
            mask,
            mx.array(-float('inf'), scaled_logits.dtype),
            scaled_logits
        )

        # Sample
        next_token = mx.random.categorical(scaled_logits, axis=-1).item()
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
