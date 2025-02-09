# src/inference/mlx/samplers/greedy.py
import mlx.core as mx
from inference.prompt_removal import remove_prompt_prefix
from inference.stop_utils import check_stop_sequences, prepare_stop_sequences

def greedy_generate(
    model,
    tokenizer,
    prompt,
    max_new_tokens=2000,
    stop_sequences=None
):
    """
    Generates text token-by-token using a purely greedy approach,
    stopping if any stop sequence appears anywhere in the decoded text.
    Then remove the prompt prefix if the final output starts with it.
    """
    if stop_sequences is None:
        stop_sequences = []
    stop_sequences = prepare_stop_sequences(stop_sequences)

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.eos_token_id]

    for _ in range(max_new_tokens):
        logits = model(mx.array(tokens, mx.uint32)[None])
        last_logits = logits[:, -1, :]
        next_token = mx.argmax(last_logits, axis=-1).item()
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
