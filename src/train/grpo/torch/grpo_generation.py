# src/train/grpo/torch/grpo_generation.py
import logging
import torch

from inference.torch.custom_generate_torch import top_p_generate_torch_with_kvcache
from train.grpo.torch.grpo_utils import gather_logprobs

logger = logging.getLogger(__name__)

def generate_single_response_and_oldlogprob(
    ref_model,
    tokenizer,
    prompt: str,
    verbose: bool = False,
    max_new_tokens: int = 2000,
    temperature: float = 0.6,
    top_p: float = 0.95
):
    """
    Generates a single response by sampling from the *reference* (frozen) model
    using top-p sampling, prepends "<think>" to the text, and then re-runs 
    the final text through 'ref_model' to compute the sum of log-probs.

    Typically, 'ref_model' is your frozen policy snapshot (old policy) in 
    PPO/GRPO approaches, ensuring that sampling is done strictly from 
    the old/frozen distribution.

    Args:
        ref_model (PreTrainedModel): A *frozen* reference policy (snapshot of the old policy).
        tokenizer (PreTrainedTokenizer): Tokenizer for 'ref_model'.
        prompt (str): Input prompt text.
        verbose (bool): If True, logs the final generated response.
        max_new_tokens (int): Maximum tokens to generate in sampling.
        temperature (float): Temperature for sampling.
        top_p (float): Top-p (nucleus) sampling threshold.

    Returns:
        response_text (str): The generated text (with "<think>" prepended).
        sum_lp (float): Sum of log probabilities (under 'ref_model') for that text.
    """
    # Sequences to stop generation
    stop_seqs = ["<|endoftext|>"]

    # 1) Perform top-p generation with ref_model
    raw_resp = top_p_generate_torch_with_kvcache(
        model=ref_model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_p=top_p,
        stop_sequences=stop_seqs
    ).strip()

    # 2) Handle empty generation
    if not raw_resp:
        logger.warning("[WARN] Generated an empty response; using <|endoftext|> as fallback.")
        raw_resp = "<|endoftext|>"

    # 3) Prepend "<think>"
    response_text = "<think>" + raw_resp

    if verbose:
        logger.info(f"Reference model response: {response_text}")

    # 4) Tokenize the final text
    tokenized_response = tokenizer(response_text, return_tensors="pt")
    if tokenized_response["input_ids"].numel() == 0:
        logger.warning("[WARN] Empty token sequence; substituting eos_token_id.")
        device = getattr(ref_model, "device", torch.device("cpu"))
        tokenized_response["input_ids"] = torch.tensor([[tokenizer.eos_token_id]], device=device)

    # Move inputs to the ref_model's device
    ref_device = getattr(ref_model, "device", torch.device("cpu"))
    tokenized_response = {k: v.to(ref_device) for k, v in tokenized_response.items()}

    # 5) Forward pass on ref_model to compute logprobs
    with torch.no_grad():
        out = ref_model(**tokenized_response)
        seq_lp = gather_logprobs(out.logits, tokenized_response["input_ids"])  # shape [1]
        sum_lp = seq_lp.item()

    return response_text, sum_lp
