# src/inference/infer.py

import logging
from typing import List, Optional

# Imports
from inference.torch.torch_chat_generation import torch_chat_generation
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
    use_chat_template: bool = False,
    stream: bool = False
):
    """
    Execute inference using a *preloaded* model/tokenizer.

    - If is_mlx=True, calls 'mlx_chat_generation'.
    - If is_mlx=False (Torch), calls 'torch_chat_generation'.
    - 'sampler' can be 'default' => greedy or 'top_p' => nucleus sampling.
    - 'use_chat_template' can enable role-based chat prompts.
    - 'stream=True/False' decides whether to do streaming or non-streaming decode.

    :param model: MLX or Torch model object
    :param tokenizer: MLX or Torch tokenizer
    :param is_mlx: True => MLX path, False => Torch path
    :param system_prompt: Optional system instruction for context
    :param user_messages: List of user messages so far
    :param assistant_messages: List of assistant messages so far
    :param max_new_tokens: Maximum tokens to generate
    :param sampler: "default" or "top_p"
    :param temperature: For top-p sampling
    :param top_p: Probability cutoff for nucleus sampling
    :param stop_sequences: If provided, generation stops as soon as text contains any
    :param use_chat_template: If True, uses chat template logic 
                              (build_chat_prompt) to create conversation prompt
    :param stream: If True, tries streaming decode in both MLX and Torch
    :return: The generated assistant response (string)
    """

    logger.debug(
        "run_inference => is_mlx=%s, sampler=%s, temp=%.2f, top_p=%.2f, "
        "use_chat_template=%s, stop=%s, stream=%s",
        is_mlx, sampler, temperature, top_p, use_chat_template, stop_sequences, stream
    )

    if is_mlx:
        # MLX chat generation (can also handle 'stream=True' if implemented)
        return mlx_chat_generation(
            loaded_model=model,
            loaded_tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            max_new_tokens=max_new_tokens,
            sampler=sampler,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            use_chat_template=use_chat_template,
            stream=stream  # pass in the stream param
        )
    else:
        # Torch chat generation (if stream=True, do text streaming w/ TextIteratorStreamer)
        return torch_chat_generation(
            loaded_model=model,
            loaded_tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            max_new_tokens=max_new_tokens,
            sampler=sampler,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            use_chat_template=use_chat_template,
            stream=stream  # pass in the stream param
        )