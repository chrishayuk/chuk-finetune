# src/train/grpo/torch/grpo_generation.py
import logging
import torch

# imports
from inference.torch.custom_generate_torch import top_p_generate_torch, top_p_generate_torch_with_kvcache
from train.grpo.torch.grpo_utils import gather_logprobs

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
    Generates a single response from 'model' using top-p sampling
    (temperature=0.6, top_p=0.95 by default). Then prepends "<think>"
    to the final text, and computes the old log-prob by re-running
    that final text through the model and calling gather_logprobs(...).
    """

    stop_seqs = ["<|endoftext|>"]

    # 1) Perform token-by-token top-p generation, matching deepseek
    raw_resp = top_p_generate_torch_with_kvcache(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=0.6,
        top_p=0.95,
        stop_sequences = stop_seqs
    ).strip()

    # 2) Add "<think>" prefix
    response_text = "<think>" + raw_resp

    if verbose:
        logger.info(f"Model: {response_text}")

    # 3) Re-run for logprob
    tokenized_response = tokenizer(response_text, return_tensors="pt")

    if tokenized_response["input_ids"].numel() == 0:
        logger.warning("[WARN] Empty token sequence; fallback to eos.")
        tokenized_response["input_ids"] = torch.tensor([[tokenizer.eos_token_id]])

    tokenized_response = {k: v.to(model.device) for k, v in tokenized_response.items()}

    with torch.no_grad():
        out = model(**tokenized_response)
        # sum up logprobs of the entire sequence
        sum_lp = gather_logprobs(out.logits, tokenized_response["input_ids"]).item()

    return response_text, sum_lp
