# src/train/grpo/mlx/custom_generate_mlx.py
import re
import mlx.core as mx

#################################################
# 0. Internal Helpers for Anywhere Stop Sequences
#################################################

def _make_stop_regex(stop_seq: str):
    """
    Given a stop sequence like 'User:', return a regex that matches it anywhere 
    in the text (not just at the end).
    Example: 'User:' => re.compile(r'User:')
    """
    escaped_seq = re.escape(stop_seq)
    return re.compile(escaped_seq)

def _check_stop_sequences(decoded_text: str, stop_sequences):
    """
    Check if 'decoded_text' contains any of the stop-sequence regexes.
    If found, we truncate the text at the *earliest* match index.

    Return the truncated text if a match is found, otherwise None.
    """
    earliest_idx = None
    for pattern, original_seq in stop_sequences:
        match = pattern.search(decoded_text)
        if match:
            idx = match.start()
            if earliest_idx is None or idx < earliest_idx:
                earliest_idx = idx

    if earliest_idx is not None:
        # return everything up to earliest_idx
        return decoded_text[:earliest_idx]

    return None

def _prepare_stop_sequences(stop_sequences):
    """
    Turn each string in 'stop_sequences' into a compiled regex
    that matches anywhere in the text.
    Returns a list of (compiled_regex, original_seq).
    """
    result = []
    for seq in stop_sequences:
        pattern = _make_stop_regex(seq)
        result.append((pattern, seq))
    return result

#################################################
# 1. Basic Generation Functions
#################################################

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
    """
    if stop_sequences is None:
        stop_sequences = []
    stop_sequences = _prepare_stop_sequences(stop_sequences)

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

        # Check stop sequences anywhere
        current_text = tokenizer.decode(tokens)
        maybe_truncated = _check_stop_sequences(current_text, stop_sequences)
        if maybe_truncated is not None:
            return maybe_truncated

    return tokenizer.decode(tokens)


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
    Generates text token-by-token using top-k sampling,
    stopping if any stop sequence appears in the decoded text.
    """
    if stop_sequences is None:
        stop_sequences = []
    stop_sequences = _prepare_stop_sequences(stop_sequences)

    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.eos_token_id]

    for _ in range(max_tokens):
        logits = model(mx.array(tokens, mx.uint32)[None])
        last_logits = logits[:, -1, :]

        # Scale by 1/temperature
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

        # Check stop sequences
        current_text = tokenizer.decode(tokens)
        maybe_truncated = _check_stop_sequences(current_text, stop_sequences)
        if maybe_truncated is not None:
            return maybe_truncated

    return tokenizer.decode(tokens)


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
    Generates text token-by-token using top-p (nucleus) sampling,
    stopping if any stop sequence appears in the decoded text.
    """
    if stop_sequences is None:
        stop_sequences = []
    stop_sequences = _prepare_stop_sequences(stop_sequences)

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

        # Sample within nucleus
        chosen_index = mx.random.categorical(mx.log(safe_probs + 1e-12), axis=-1).item()
        chosen_index_arr = mx.array(chosen_index, mx.uint32).reshape((1, 1))
        next_token = mx.take_along_axis(sorted_indices, chosen_index_arr, axis=-1).item()

        tokens.append(next_token)

        # Check EOS
        if next_token == tokenizer.eos_token_id:
            break

        # Check stop sequences anywhere
        current_text = tokenizer.decode(tokens)
        maybe_truncated = _check_stop_sequences(current_text, stop_sequences)
        if maybe_truncated is not None:
            return maybe_truncated

    return tokenizer.decode(tokens)

#################################################
# 2. Multi-sample Generation Helper
#################################################

def top_p_sample_n(
    model,
    tokenizer,
    prompt,
    n=4,
    max_tokens=2000,
    temperature=0.6,
    top_p=0.95
):
    """
    Generates 'n' independent samples using top-p sampling
    with default temp=0.6 and top_p=0.95.
    """
    samples = []
    for _ in range(n):
        gen = top_p_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature
        )
        samples.append(gen)
    return samples


#################################################
# 3. pass@k (pass@1) and Consensus Evaluation
#################################################

def is_correct(generated_text, reference):
    """
    Basic correctness check. Replace with logic suitable
    for your domain (exact match, numeric parse, etc.).
    """
    return generated_text.strip() == reference.strip()


def evaluate_pass1(
    model,
    tokenizer,
    prompt,
    reference,
    k=4,
    max_tokens=2000,
    temperature=0.6,
    top_p=0.95
):
    """
    Computes pass@1 for a single (prompt, reference) pair.
    - Generate 'k' samples
    - Evaluate correctness of each
    - Return fraction correct (pass@1).
    """
    correct_count = 0
    for _ in range(k):
        gen = top_p_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        if is_correct(gen, reference):
            correct_count += 1
    return correct_count / k


def evaluate_dataset_pass1(
    model,
    tokenizer,
    questions,
    references,
    k=4,
    max_tokens=2000,
    temperature=0.6,
    top_p=0.95
):
    """
    Computes pass@1 across a dataset of (question, reference) pairs
    and returns the average pass@1.
    """
    scores = []
    for q, ref in zip(questions, references):
        s = evaluate_pass1(
            model=model,
            tokenizer=tokenizer,
            prompt=q,
            reference=ref,
            k=k,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        scores.append(s)
    return sum(scores) / len(scores)


def evaluate_consensus(
    model,
    tokenizer,
    prompt,
    reference,
    n_samples=64,
    max_tokens=2000,
    temperature=0.6,
    top_p=0.95
):
    """
    Majority-vote consensus for a single prompt.
    - Generate 'n_samples' responses
    - Convert each to a discrete answer
    - Take majority vote
    - Return 1 if majority is correct, else 0
    """
    from collections import Counter
    completions = []
    for _ in range(n_samples):
        gen = top_p_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        completions.append(gen.strip())

    # Count occurrences of each unique completion
    counter = Counter(completions)
    majority_answer, _ = counter.most_common(1)[0]

    return 1.0 if is_correct(majority_answer, reference) else 0.0


def evaluate_dataset_consensus(
    model,
    tokenizer,
    questions,
    references,
    n_samples=64,
    max_tokens=2000,
    temperature=0.6,
    top_p=0.95
):
    """
    Evaluate consensus@64 across a dataset of Q&A pairs.
    Returns fraction correct by majority vote.
    """
    correct_count = 0
    for q, r in zip(questions, references):
        val = evaluate_consensus(
            model=model,
            tokenizer=tokenizer,
            prompt=q,
            reference=r,
            n_samples=n_samples,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p
        )
        correct_count += val
    return correct_count / len(questions)
