import mlx.core as mx
from inference.stop_utils import check_stop_sequences, prepare_stop_sequences
from inference.prompt_removal import remove_prompt_prefix
from inference.mlx.kv_cache import KVCache

def top_p_generate_with_kvcache(
    model,
    tokenizer,
    prompt: str,
    max_tokens: int = 200,
    top_p: float = 0.9,
    temperature: float = 1.0,
    stop_sequences=None,
    cache=None,
):
    """
    Token-by-token top-p (ASCENDING) sampling using a QWen-style KV-cache,
    but we pass the entire [prompt + generated tokens] each time.

    Why do this? Because some QWen variants:
      - Don't accept a `start_pos` argument
      - Don't handle "partial feed" automatically
      - But DO keep an offset in the in-place KV-cache so repeated tokens are
        skipped internally.

    Steps:
      1) Encode the full prompt => tokens
      2) Feed all tokens once (filling the cache)
      3) On each new generation step, sample from the last logits -> next_token
      4) Append `next_token` to `tokens`
      5) Re-send the entire `tokens` to the model again
      6) The model sees in its KV-cache that the first len(tokens)-1 are already processed,
         so it only processes the new token(s).

    This approach is slower than pure partial feed, but it usually still 
    avoids re-computing everything from scratch if the model actually 
    respects its in-place KV-cache.

    Args:
        model: A QWen model that updates the KV-cache in place (no `start_pos` param).
        tokenizer: Has `.encode()` / `.decode()`.
        prompt (str): Initial prompt text.
        max_tokens (int): Max new tokens to generate.
        top_p (float): Nucleus sampling threshold.
        temperature (float): Softmax temperature.
        stop_sequences (List[str]): If these appear in text, stop generation.
        cache (List[KVCache], optional): If you have a pre-made list-of-caches, pass it;
          else we'll create a new one.

    Returns:
        Final decoded text (minus the original prompt).
    """
    if stop_sequences is None:
        stop_sequences = []
    stop_sequences = prepare_stop_sequences(stop_sequences)

    # 1) Encode the prompt
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.eos_token_id]

    # 2) Create or reuse a per-layer KV cache
    if cache is None:
        num_layers = getattr(model, "num_layers", 24)
        cache = [KVCache() for _ in range(num_layers)]

    # 3) First pass: feed entire prompt
    input_arr = mx.array([tokens], dtype=mx.uint32)
    logits = model(input_arr, cache=cache)  # shape: [1, len(tokens), vocab_size]

    # 4) Generate up to max_tokens
    for _ in range(max_tokens):
        # a) Get logits for the last token
        last_logits = logits[:, -1, :]
        scaled_logits = last_logits / temperature

        # b) Convert to probabilities
        probs = mx.softmax(scaled_logits, axis=-1)

        # c) Ascending sort
        sorted_indices = mx.argsort(probs, axis=-1)
        sorted_probs   = mx.take_along_axis(probs, sorted_indices, axis=-1)

        # d) Cumulative sum => identify top-p tail
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)
        threshold_mask   = cumulative_probs > (1.0 - top_p)
        truncated_probs  = mx.where(threshold_mask, sorted_probs, mx.array(0.0, probs.dtype))

        # e) Re-normalize
        sum_trunc = truncated_probs.sum(axis=-1, keepdims=True) + 1e-12
        truncated_probs = truncated_probs / sum_trunc

        # f) Sample from truncated distribution
        chosen_idx = mx.random.categorical(mx.log(truncated_probs + 1e-12), axis=-1).item()
        chosen_idx_arr = mx.array(chosen_idx, mx.uint32).reshape((1, 1))
        next_token = mx.take_along_axis(sorted_indices, chosen_idx_arr, axis=-1).item()

        tokens.append(next_token)

        # Check EOS or stop sequences
        if next_token == tokenizer.eos_token_id:
            break
        current_text = tokenizer.decode(tokens)
        maybe_truncated = check_stop_sequences(current_text, stop_sequences)
        if maybe_truncated is not None:
            return maybe_truncated

        # g) Re-feed the entire tokens array
        #    The model sees the newly appended token, but in theory 
        #    it can skip old tokens due to KV-cache in-place offset. 
        input_arr = mx.array([tokens], dtype=mx.uint32)
        logits = model(input_arr, cache=cache)  # shape: [1, len(tokens), vocab_size]

    # 5) Return final text, minus the original prompt
    raw_output = tokenizer.decode(tokens)
    return remove_prompt_prefix(raw_output, prompt)
