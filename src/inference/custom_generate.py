# src/inference/custom_generate.py
import logging

# imports
from inference.torch.custom_generate_torch import top_p_generate_torch
from inference.mlx.custom_generate_mlx import top_p_generate as top_p_generate_mlx

# logging
logger = logging.getLogger(__name__)

def custom_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int,
    temperature: float,
    top_p: float,
    is_mlx: bool,
    stop_sequences=None
):
    """
    Unified custom generation that dispatches to MLX or Torch top-p.
    Now accepts an optional stop_sequences list.
    """
    # check for stop sequences
    if stop_sequences is None:
        stop_sequences = []

    # log
    logger.info("custom_generate called (is_mlx=%s, top_p=%.2f, temp=%.2f, stop_seqs=%s)",
                is_mlx, top_p, temperature, stop_sequences)

    if is_mlx:
        # Make sure mlx_top_p also accepts stop_sequences
        return top_p_generate_mlx(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences
        )
    else:
        # Make sure top_p_generate_torch also accepts stop_sequences
        return top_p_generate_torch(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences
        )
