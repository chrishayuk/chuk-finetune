# src/inference/mlx/evaluation.py
from inference.mlx.samplers.top_p import top_p_generate

def is_correct(generated_text: str, reference: str) -> bool:
    """
    Basic correctness check. Replace with logic suitable
    for your domain (exact string match, numeric parse, etc.).
    """
    return generated_text.strip() == reference.strip()

def evaluate_pass1(
    model,
    tokenizer,
    prompt: str,
    reference: str,
    k: int = 4,
    max_tokens: int = 2000,
    temperature: float = 0.6,
    top_p: float = 0.95
) -> float:
    """
    Computes pass@1 for a single (prompt, reference) pair.
    1) Generate 'k' samples using top_p_generate
    2) Evaluate correctness of each
    3) Return fraction correct (pass@1).
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
    questions: list,
    references: list,
    k: int = 4,
    max_tokens: int = 2000,
    temperature: float = 0.6,
    top_p: float = 0.95
) -> float:
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
    prompt: str,
    reference: str,
    n_samples: int = 64,
    max_tokens: int = 2000,
    temperature: float = 0.6,
    top_p: float = 0.95
) -> float:
    """
    Majority-vote consensus for a single prompt.
    1) Generate 'n_samples' responses
    2) Convert each to a discrete answer
    3) Take majority vote
    4) Return 1 if majority is correct, else 0
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
    questions: list,
    references: list,
    n_samples: int = 64,
    max_tokens: int = 2000,
    temperature: float = 0.6,
    top_p: float = 0.95
) -> float:
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
