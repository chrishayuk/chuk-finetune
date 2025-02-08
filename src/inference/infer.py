# src/inference/infer.py

import logging
from typing import List, Optional

# Our new Torch chat code that supports step-by-step generation (greedy or top-p)
from inference.torch.torch_chat_generation import torch_chat_generation

# Our MLX chat code that supports streaming greedy or top-p, plus an optional chat template
from inference.mlx.mlx_chat_generation import mlx_chat_generation

logger = logging.getLogger(__name__)

def run_inference(
    model,
    tokenizer,
    is_mlx: bool,
    system_prompt: str,
    user_messages: List[str],
    assistant_messages: List[str],
    max_new_tokens: int,
    sampler: str = "default",
    temperature: float = 0.6,
    top_p: float = 0.95,
    stop_sequences: Optional[List[str]] = None,
    use_chat_template: bool = False
):
    """
    Execute inference using a *preloaded* model/tokenizer.

    - If is_mlx=True, calls 'mlx_chat_generation' with either streaming GREEDY 
      or top-p from your custom code. Optionally, it can use a role-based chat template 
      if 'use_chat_template=True'.
    - If is_mlx=False (Torch), calls 'torch_chat_generation' with either 
      step-by-step GREEDY ('default') or top-p sampling.
    
    This ensures 'default' yields more complete, chatty answers for both MLX 
    and Torch, and 'top_p' uses your nucleus sampling with immediate stop 
    if provided.

    :param model: MLX or Torch model object
    :param tokenizer: MLX or Torch tokenizer
    :param is_mlx: True => MLX path, False => Torch
    :param system_prompt: Optional system instruction for context
    :param user_messages: List of user messages so far
    :param assistant_messages: List of assistant messages so far
    :param max_new_tokens: Maximum tokens to generate
    :param sampler: "default" or "top_p"
    :param temperature: For top-p sampling
    :param top_p: Probability cutoff for nucleus sampling
    :param stop_sequences: If provided, generation stops as soon as text contains any
    :param use_chat_template: If True, uses the chat template logic 
                              (build_chat_prompt) to create the conversation 
                              prompt in both Torch and MLX.
    :return: The generated assistant response (string)
    """

    logger.debug(
        "run_inference => sampler=%s, temp=%.2f, top_p=%.2f, use_chat_template=%s, stop=%s",
        sampler, temperature, top_p, use_chat_template, stop_sequences
    )

    if sampler == "top_p":
        logger.debug("Using top-p sampling (temperature=%.2f, top_p=%.2f).", temperature, top_p)

        if is_mlx:
            # MLX => call 'mlx_chat_generation' with sampler='top_p'
            return mlx_chat_generation(
                loaded_model=model,
                loaded_tokenizer=tokenizer,
                system_prompt=system_prompt,
                user_messages=user_messages,
                assistant_messages=assistant_messages,
                max_new_tokens=max_new_tokens,
                sampler="top_p",
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences,
                use_chat_template=use_chat_template
            )
        else:
            # Torch => call 'torch_chat_generation' with sampler='top_p'
            return torch_chat_generation(
                loaded_model=model,
                loaded_tokenizer=tokenizer,
                system_prompt=system_prompt,
                user_messages=user_messages,
                assistant_messages=assistant_messages,
                max_new_tokens=max_new_tokens,
                sampler="top_p",
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences,
                use_chat_template=use_chat_template
            )

    else:
        # debug
        logger.debug("Using '%s' sampler => default chat logic.", sampler)

        if is_mlx:
            # MLX => streaming GREEDY
            return mlx_chat_generation(
                loaded_model=model,
                loaded_tokenizer=tokenizer,
                system_prompt=system_prompt,
                user_messages=user_messages,
                assistant_messages=assistant_messages,
                max_new_tokens=max_new_tokens,
                sampler="default",
                stop_sequences=stop_sequences,
                use_chat_template=use_chat_template
            )
        else:
            # Torch => step-by-step GREEDY
            return torch_chat_generation(
                loaded_model=model,
                loaded_tokenizer=tokenizer,
                system_prompt=system_prompt,
                user_messages=user_messages,
                assistant_messages=assistant_messages,
                max_new_tokens=max_new_tokens,
                sampler="default",
                stop_sequences=stop_sequences,
                use_chat_template=use_chat_template
            )