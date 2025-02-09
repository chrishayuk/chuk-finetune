# src/train/grpo/mlx/grpo_generation.py

import logging
import mlx.core as mx

# imports
from inference.mlx.custom_generate_mlx import top_p_generate
from train.grpo.mlx.grpo_loss import gather_logprobs

# logger
logger = logging.getLogger(__name__)

def generate_single_response_and_oldlogprob(
    model,
    tokenizer,
    prompt: str,
    verbose: bool = False,
    max_new_tokens: int = 2048
):
    """
    Generates a single response from 'model' using Top-p (nucleus) sampling 
    and computes the 'old log-prob' as negative log-likelihood for that response.
    
    By default, we use:
      - temperature=0.6
      - top_p=0.95
    
    If you want more or less randomness, adjust these parameters.
    """

    stop_seqs = ["<|endoftext|>"]

    # 1) Perform token-by-token top-p generation, matching deepseek
    response_text = top_p_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_new_tokens,
        temperature=0.6,
        top_p=0.95,
        stop_sequences = stop_seqs
    ).strip()

    # Optionally prepend a "thinking" token
    response_text = "<think>" + response_text

    # --- (2) Log if verbose
    if verbose:
        logger.info(f"Model: {response_text}")

    # --- (3) Tokenize the response
    tokens = tokenizer.encode(response_text)
    if not tokens:
        logger.warning("[WARN] Empty token sequence; fallback to eos.")
        tokens = [tokenizer.eos_token_id]

    # --- (4) Forward pass to get logits
    logits = model(mx.array(tokens, mx.uint32)[None])

    # --- (5) Gather negative log-likelihood (sum of log-probs)
    sum_lp = float(gather_logprobs(logits, tokens))

    # Return the text plus the log-prob
    return response_text, sum_lp