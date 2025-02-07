# src/train/grpo/torch/grpo_generation.py
import logging
import torch

# imports
from train.grpo.torch.grpo_utils import gather_logprobs
from inference.torch.custom_generate_torch import greedy_generate_torch

# logger
logger = logging.getLogger(__name__)

def generate_single_response_and_oldlogprob(
    model,
    tokenizer,
    prompt: str,
    verbose: bool = False,
    max_new_tokens: int = 2000
):
    """
    Generates a single response from 'model' using a manual token-by-token greedy approach
    that closely matches MLX's `greedy_generate(...).`

    Then prepends "<think>" to the final text, and computes the old log-prob 
    by re-running the final text through the model and calling gather_logprobs(...).
    """

    # 1) Perform token-by-token greedy generation (like MLX)
    raw_resp = greedy_generate_torch(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens
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
