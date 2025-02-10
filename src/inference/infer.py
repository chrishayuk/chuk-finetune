# src/inference/infer.py
import logging
from typing import List, Optional, Union

# logger
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
    stream: bool = False,
    num_responses: int = 1  # NEW
) -> Union[str, List[str]]:
    """
    Execute inference using a *preloaded* model/tokenizer.
    - If num_responses > 1 and sampler=='top_p', we produce multiple top-p samples
      by calling the underlying chat_generation multiple times.
    - Otherwise, we produce a single string as before.

    Returns:
      - A single string (the final generation),
      OR
      - A list of strings if multiple responses were generated.
    """

    logger.debug(
        "run_inference => is_mlx=%s, sampler=%s, temp=%.2f, top_p=%.2f, "
        "use_chat_template=%s, stop=%s, stream=%s, num_responses=%d",
        is_mlx, sampler, temperature, top_p, use_chat_template,
        stop_sequences, stream, num_responses
    )

    # 1) If the user wants multiple responses (num_responses>1) AND
    #    sampler == "top_p", we do multiple calls:
    if num_responses > 1 and sampler == "top_p":
        # We'll do multiple calls to the underlying chat_generation function,
        # each returning a single string, then gather them in a list.

        # Check that we're not in 'chat' mode with streaming, etc. 
        # (You can decide how to handle that. We'll just do the loop.)
        responses = []
        for _ in range(num_responses):
            single_resp = _run_single_inference(
                model=model,
                tokenizer=tokenizer,
                is_mlx=is_mlx,
                system_prompt=system_prompt,
                user_messages=user_messages,
                assistant_messages=assistant_messages,
                max_new_tokens=max_new_tokens,
                sampler=sampler,
                temperature=temperature,
                top_p=top_p,
                stop_sequences=stop_sequences,
                use_chat_template=use_chat_template,
                stream=stream
            )
            responses.append(single_resp)
        return responses

    else:
        # 2) Single response path
        return _run_single_inference(
            model=model,
            tokenizer=tokenizer,
            is_mlx=is_mlx,
            system_prompt=system_prompt,
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            max_new_tokens=max_new_tokens,
            sampler=sampler,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences,
            use_chat_template=use_chat_template,
            stream=stream
        )

def _run_single_inference(
    model,
    tokenizer,
    is_mlx: bool,
    system_prompt: str,
    user_messages: List[str],
    assistant_messages: List[str],
    max_new_tokens: int,
    sampler: str,
    temperature: float,
    top_p: float,
    stop_sequences: Optional[List[str]],
    use_chat_template: bool,
    stream: bool
) -> str:
    """
    Internal helper that does exactly one call to either MLX or Torch chat generation,
    returning a single final string. This code was previously in run_inference.
    """
    if is_mlx:
        # mlx
        #from inference.mlx.mlx_chat_generation_native import mlx_chat_generation
        from inference.mlx.mlx_chat_generation import mlx_chat_generation

        # MLX chat generation
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
            stream=stream
        )
    else:
        # torch
        from inference.torch.torch_chat_generation import torch_chat_generation

        # Torch chat generation
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
            stream=stream
        )