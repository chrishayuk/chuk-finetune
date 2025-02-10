# src/train/grpo/torch/grpo_generation.py
import logging
import torch

from inference.torch.custom_generate_torch import (
    top_p_generate_torch,
    top_p_generate_torch_with_kvcache
)
from train.grpo.torch.grpo_utils import gather_logprobs

# logger
logger = logging.getLogger(__name__)

def generate_single_response_and_oldlogprob(
    model,
    tokenizer,
    prompt: str,
    verbose: bool = False,
    max_new_tokens: int = 2000,
    temperature: float = 0.6,
    top_p: float = 0.95
):
    """
    Generates a single response from 'model' using top-p sampling.
    Then, prepends "<think>" to the final text, and computes the old log-prob 
    by re-running that final text through the model and calling gather_logprobs(...).

    Args:
        model (PreTrainedModel): A HuggingFace-like model with a .generate or custom generation method.
        tokenizer (PreTrainedTokenizer): Corresponding tokenizer for `model`.
        prompt (str): The input prompt text.
        verbose (bool): If True, logs the final generated response.
        max_new_tokens (int): Maximum tokens to generate.
        temperature (float): Sampling temperature.
        top_p (float): Top-p sampling threshold.

    Returns:
        response_text (str): The generated response (with "<think>" prepended).
        sum_lp (float): Sum of log probabilities for all tokens in `response_text`.
    """

    # Sequences to stop generation
    stop_seqs = ["<|endoftext|>"]

    # 1) Perform top-p generation with kv-cache for efficiency
    raw_resp = top_p_generate_torch_with_kvcache(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_sequences=stop_seqs
    ).strip()

    # 2) Handle the case where generation might be empty
    if not raw_resp:
        logger.warning("[WARN] Generated an empty response; using <|endoftext|> as fallback.")
        raw_resp = "<|endoftext|>"

    # 3) Prepend "<think>"
    response_text = "<think>" + raw_resp

    if verbose:
        logger.info(f"Model response: {response_text}")

    # 4) Re-run for logprob
    tokenized_response = tokenizer(response_text, return_tensors="pt")

    # Check for zero-length token sequence
    if tokenized_response["input_ids"].numel() == 0:
        logger.warning("[WARN] Empty token sequence; substituting eos_token_id.")
        tokenized_response["input_ids"] = torch.tensor([[tokenizer.eos_token_id]], device=model.device)

    # Move inputs to correct device
    tokenized_response = {k: v.to(model.device) for k, v in tokenized_response.items()}

    # 5) Forward pass to compute logprobs
    with torch.no_grad():
        out = model(**tokenized_response)
        # sum up logprobs of the entire sequence
        seq_lp = gather_logprobs(out.logits, tokenized_response["input_ids"])  # shape [B], B=1
        sum_lp = seq_lp.item()  # Convert tensor scalar -> Python float

    return response_text, sum_lp