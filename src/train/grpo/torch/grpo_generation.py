# src/train/grpo/torch/grpo_generation.py
import logging
import torch

from train.grpo.torch.grpo_utils import gather_logprobs
from inference.torch.custom_generate_torch import greedy_generate_torch

logger = logging.getLogger(__name__)

def generate_single_response_and_oldlogprob(
    model,
    tokenizer,
    prompt: str,
    verbose: bool = False,
    max_tokens: int = 2000
):
    """
    Generates a single response from 'model' using a manual token-by-token greedy approach
    that closely matches MLX's `greedy_generate(...).`

    Then prepends "<think>" to the final text, and computes the old log-prob 
    by re-running the final text through the model and calling gather_logprobs(...).

    Args:
        model: A Hugging Face (or compatible) Torch model that can return logits 
               of shape [1, seq_len, vocab_size].
        tokenizer: The tokenizer for encoding/decoding text.
        prompt (str): The text prompt to feed in.
        verbose (bool): If True, logs the final response text.
        max_tokens (int): The maximum number of tokens to generate 
                          beyond the initial prompt.

    Returns:
        (response_text, sum_lp):
            response_text: The final text, prefixed with "<think>".
            sum_lp: A float of the summed logprob for that final text.
    """

    # 1) Perform token-by-token greedy generation (like MLX)
    raw_resp = greedy_generate_torch(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=max_tokens
    ).strip()

    # 2) Add "<think>" prefix
    response_text = "<think>" + raw_resp

    if verbose:
        logger.info(f"Model: {response_text}")

    # 3) Re-run for logprob
    tokenized_response = tokenizer(response_text, return_tensors="pt")
    
    # fallback if empty
    if tokenized_response["input_ids"].numel() == 0:
        logger.warning("[WARN] Empty token sequence; fallback to eos.")
        tokenized_response["input_ids"] = torch.tensor([[tokenizer.eos_token_id]])

    tokenized_response = {k: v.to(model.device) for k, v in tokenized_response.items()}

    with torch.no_grad():
        out = model(**tokenized_response)
        # sum up logprobs of the entire sequence
        sum_lp = gather_logprobs(out.logits, tokenized_response["input_ids"]).item()

    return response_text, sum_lp
