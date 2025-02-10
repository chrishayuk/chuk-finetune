# src/train/grpo/mlx/grpo_generation.py
import logging
import mlx.core as mx

from inference.mlx.custom_generate_mlx import top_p_generate
from train.grpo.mlx.grpo_loss import gather_logprobs

logger = logging.getLogger(__name__)

def generate_single_response_and_oldlogprob(
    ref_model,
    tokenizer,
    prompt: str,
    verbose: bool = False,
    max_new_tokens: int = 2048,
    temperature: float = 0.6,
    top_p: float = 0.95
):
    """
    Generates a single response by sampling from the *reference* (frozen) MLX model
    using top-p (nucleus) sampling, prepends "<think>" to the final text,
    and computes the sum of log-probs for the generated text.

    In GRPO/PPO, 'ref_model' typically refers to the old/frozen policy.

    Args:
        ref_model: An MLX model object (a frozen snapshot of your old policy).
        tokenizer: A tokenizer with `.encode()` returning integer token IDs.
        prompt (str): Prompt text to feed to the ref_model.
        verbose (bool): If True, logs the final generated response.
        max_new_tokens (int): Maximum new tokens to generate.
        temperature (float): Sampling temperature (higher => more random).
        top_p (float): Top-p nucleus threshold (lower => more conservative sampling).

    Returns:
        response_text (str): Generated text (with "<think>" prepended).
        sum_lp (float): Sum of log-probabilities for the entire sequence (under ref_model).
    """
    stop_seqs = ["<|endoftext|>"]

    # 1) Generate the response text from the reference model
    raw_resp = top_p_generate(
        model=ref_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_sequences=stop_seqs
    ).strip()

    # 2) Fallback if empty
    if not raw_resp:
        logger.warning("[WARN] Generated an empty response; using <|endoftext|> as fallback.")
        raw_resp = "<|endoftext|>"

    # 3) Prepend "<think>" to the final text
    response_text = "<think>" + raw_resp

    # 4) Verbose logging
    if verbose:
        logger.info(f"Reference model response: {response_text}")

    # 5) Encode + re-run text to compute log-prob
    tokens = tokenizer.encode(response_text)
    if not tokens:
        logger.warning("[WARN] Empty token sequence; fallback to eos.")
        tokens = [tokenizer.eos_token_id]

    # Forward pass for logits
    tokens_mlx = mx.array(tokens, mx.uint32)[None]  # shape [1, seq_len]
    logits = ref_model(tokens_mlx)  # => shape [1, seq_len, vocab_size]

    # Gather log-probs
    sum_lp = float(gather_logprobs(logits, tokens))  # shape [1], convert to float

    return response_text, sum_lp
