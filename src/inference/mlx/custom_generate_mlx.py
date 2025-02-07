# src/train/grpo/mlx/custom_generate_mlx.py
import mlx.core as mx

def greedy_generate(model, tokenizer, prompt, max_new_tokens=2000):
    """
    Generates text token-by-token using a purely greedy approach 
    (i.e., always picking argmax of the logits).

    :param model: The MLX model capable of returning logits of shape [1, seq_len, vocab_size]
    :param tokenizer: The tokenizer to encode/decode the text
    :param prompt: The text prompt to start generation from
    :param max_tokens: The maximum number of additional tokens to generate
    :return: The decoded string of tokens (prompt + generated content)
    """
    # Encode the prompt
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.eos_token_id]

    # Iteratively generate up to max_tokens
    for _ in range(max_new_tokens):
        # Forward pass: shape [1, current_seq_len, vocab_size]
        logits = model(mx.array(tokens, mx.uint32)[None])

        # Take the last position's logits => shape [1, vocab_size]
        last_logits = logits[:, -1, :]

        # Argmax for greedy
        next_token = mx.argmax(last_logits, axis=-1).item()

        # Append
        tokens.append(next_token)

        # Optionally break if we encounter EOS
        if next_token == tokenizer.eos_token_id:
            break

    # Decode back to text
    return tokenizer.decode(tokens)


def top_k_generate(model, tokenizer, prompt, max_tokens=200, top_k=5, temperature=1.0):
    """
    Generates text token-by-token using top-k sampling.
    """
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.eos_token_id]

    for _ in range(max_tokens):
        logits = model(mx.array(tokens, mx.uint32)[None])
        last_logits = logits[:, -1, :]

        # Scale by 1/temperature if desired
        scaled_logits = last_logits * (1.0 / temperature)

        # Find the top-k token indices and mask out the rest
        kth_val = mx.topk(scaled_logits, k=top_k, axis=-1, largest=True, sorted=False)["values"][:, -1]
        mask = scaled_logits < kth_val
        scaled_logits = mx.where(mask, mx.array(-float('inf'), scaled_logits.dtype), scaled_logits)

        # Sample from top-k
        next_token = mx.random.categorical(scaled_logits, axis=-1).item()
        tokens.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(tokens)


def top_p_generate(model, tokenizer, prompt, max_tokens=200, top_p=0.9, temperature=1.0):
    """
    Generates text token-by-token using top-p (nucleus) sampling.
    """
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.eos_token_id]

    for _ in range(max_tokens):
        logits = model(mx.array(tokens, mx.uint32)[None])
        last_logits = logits[:, -1, :]
        scaled_logits = last_logits * (1.0 / temperature)

        # Convert to probabilities
        probs = mx.softmax(scaled_logits, axis=-1)

        # Sort in ascending order by probability
        sorted_indices = mx.argsort(probs, axis=-1)
        sorted_probs = mx.take_along_axis(probs, sorted_indices, axis=-1)

        # Compute cumulative probabilities from the least likely to the most
        cumulative_probs = mx.cumsum(sorted_probs, axis=-1)

        # We want the smallest region that sums to (1 - top_p) from the top
        # so we find the cutoff in ascending order
        threshold_mask = cumulative_probs > (1.0 - top_p)
        # "Zero out" everything below the threshold
        safe_probs = mx.where(threshold_mask, sorted_probs, mx.array(0.0, probs.dtype))
        
        # Sample from the truncated distribution
        chosen_index = mx.random.categorical(mx.log(safe_probs + 1e-12), axis=-1).item()
        # Map back to the original token ID
        next_token = mx.take_along_axis(sorted_indices, mx.array(chosen_index)[None], axis=-1).item()

        tokens.append(next_token)

        if next_token == tokenizer.eos_token_id:
            break

    return tokenizer.decode(tokens)
