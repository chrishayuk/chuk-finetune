# src/train/grpo/mlx/grpo_generation.py
import logging
import mlx.core as mx

# imports
from inference.mlx.custom_generate_mlx import greedy_generate
from train.grpo.mlx.grpo_loss import gather_logprobs

# logging
logger = logging.getLogger(__name__)

def generate_single_response_and_oldlogprob(
    model,
    tokenizer,
    prompt: str,
    verbose: bool = False,
    max_new_tokens: int = 2048
):
    """
    Generates a single response from 'model' using greedy_generate
    and computes the old log-prob as negative log-likelihood for that response.
    """

    # calls inference
    response_text = greedy_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens = max_new_tokens
    ).strip()

    # We're going to force thinking
    response_text = "<think>" + response_text

    # check if verbose
    if verbose:
        # log the model response
        logger.info(f"Model: {response_text}")

    # encode the tokenize from the response
    tokens = tokenizer.encode(response_text)

    # no response from model
    if not tokens:
        logger.warning("[WARN] Empty token sequence; fallback to eos.")
        tokens = [tokenizer.eos_token_id]

    # get the logits
    logits = model(mx.array(tokens, mx.uint32)[None])

    # get the probababilities
    sum_lp = float(gather_logprobs(logits, tokens))

    # return the response, and the logit probs
    return response_text, sum_lp
