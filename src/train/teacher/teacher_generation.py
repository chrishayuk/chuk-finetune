import logging
import torch

# MLX import
import mlx.core as mx

# Torch top-p generation
from inference.torch.custom_generate_torch import top_p_generate_torch_with_kvcache
# MLX top-p generation
from inference.mlx.custom_generate_mlx import top_p_generate

# Torch log-prob gather
from train.grpo.torch.grpo_utils import gather_logprobs as gather_logprobs_torch
# MLX log-prob gather
from train.grpo.mlx.grpo_loss import gather_logprobs as gather_logprobs_mlx

logger = logging.getLogger(__name__)

def is_mlx_model(model) -> bool:
    """
    Checks if the given 'model' is an MLX model vs. a Torch model.
    One approach:
      - MLX models might have 'model.__class__.__module__' that includes 'mlx'
      - or we can store model._is_mlx = True 
    Adjust logic as needed.
    """
    # A simple check if the model has an 'mlx' attribute or internal property:
    # Or if model class is from 'mlx' library
    return hasattr(model, 'freeze') and hasattr(model, 'parameters') is False

def generate_single_teacher_response(
    teacher_model,
    tokenizer,
    prompt: str,
    verbose: bool = False,
    max_new_tokens: int = 1024,
    temperature: float = 0.7,
    top_p: float = 0.95
):
    """
    A unified teacher generation function that detects if 'teacher_model' is
    Torch or MLX, does top-p sampling, and re-runs the text to compute sum of log-probs.

    If Torch:
      - uses 'top_p_generate_torch_with_kvcache' and gather_logprobs_torch.
    If MLX:
      - uses 'top_p_generate' (MLX) and gather_logprobs_mlx.

    The logic is similar to your PPO reference generation, but we skip any 
    special tokens like "<think>" unless you specifically want them.

    Args:
        teacher_model: A Torch or MLX model object for teacher usage.
        tokenizer: A tokenizer that either belongs to Torch or MLX (but 
                   you typically know which).
        prompt (str): The input text prompt.
        verbose (bool): If True => logs final text.
        max_new_tokens (int): limit for generation.
        temperature (float): top-p sampling temperature.
        top_p (float): nucleus sampling threshold.

    Returns:
        (response_text, sum_lp):
            response_text => The generated string from teacher.
            sum_lp => sum of log probabilities for that entire output (under teacher_model).
    """
    if is_mlx_model(teacher_model):
        # =============== MLX Path ===============
        stop_seqs = ["<|endoftext|>"]

        # 1) top_p generation in MLX
        raw_resp = top_p_generate(
            model=teacher_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_seqs
        ).strip()

        if not raw_resp:
            logger.warning("[WARN] MLX teacher => empty => fallback <|endoftext|>.")
            raw_resp = "<|endoftext|>"

        response_text = raw_resp
        if verbose:
            logger.info(f"[Teacher MLX] Generated: {response_text}")

        # 2) re-run => gather log-probs
        tokens = tokenizer.encode(response_text)
        if not tokens:
            tokens = [tokenizer.eos_token_id]
        tokens_mlx = mx.array(tokens, mx.uint32)[None]  # shape [1, seq_len]
        logits = teacher_model(tokens_mlx)
        sum_lp = float(gather_logprobs_mlx(logits, tokens))

    else:
        # =============== Torch Path ===============
        stop_seqs = ["<|endoftext|>"]

        raw_resp = top_p_generate_torch_with_kvcache(
            model=teacher_model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_seqs
        ).strip()

        if not raw_resp:
            logger.warning("[WARN] Torch teacher => empty => fallback <|endoftext|>.")
            raw_resp = "<|endoftext|>"

        response_text = raw_resp
        if verbose:
            logger.info(f"[Teacher Torch] Generated: {response_text}")

        # 2) re-run => gather log-probs
        tokenized_response = tokenizer(response_text, return_tensors="pt")
        if tokenized_response["input_ids"].numel() == 0:
            device = getattr(teacher_model, "device", torch.device("cpu"))
            tokenized_response["input_ids"] = torch.tensor([[tokenizer.eos_token_id]], device=device)

        teacher_device = getattr(teacher_model, "device", torch.device("cpu"))
        tokenized_response = {k: v.to(teacher_device) for k, v in tokenized_response.items()}

        with torch.no_grad():
            out = teacher_model(**tokenized_response)
            seq_lp = gather_logprobs_torch(out.logits, tokenized_response["input_ids"])
            sum_lp = seq_lp.item()

    return response_text, sum_lp
