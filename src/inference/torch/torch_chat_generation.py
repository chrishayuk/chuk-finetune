# src/inference/torch/torch_chat_generation.py

import logging

# We'll reuse the same function for removing special tokens, 
# just as we do in MLX code:
from inference.mlx.mlx_chat_generation import remove_special_tokens_from_text

# We'll import the updated custom Torch functions 
# that accept stop_sequences:
from inference.torch.custom_generate_torch import (
    greedy_generate_torch,
    top_p_generate_torch
)

# We'll import the same chat template function you use in MLX
from inference.chat_template import build_chat_prompt

logger = logging.getLogger(__name__)

def torch_chat_generation(
    loaded_model,
    loaded_tokenizer,
    system_prompt: str,
    user_messages: list,
    assistant_messages: list,
    max_new_tokens: int,
    sampler: str = "default",
    temperature: float = 0.6,
    top_p: float = 0.95,
    stop_sequences=None,
    use_chat_template: bool = False
):
    """
    A Torch-based chat generation function that parallels MLX logic:

    1) Build the conversation prompt using 'build_chat_prompt' 
       (which can either use a built-in chat_template or fallback).
    2) If sampler == 'default', do token-by-token greedy generation 
       with stop sequences ANYWHERE.
       If sampler == 'top_p', do token-by-token top-p generation,
       also with stop sequences ANYWHERE.
    3) We remove <|...|> tokens from final text for cleanliness.
    4) Return the final cleaned text.

    :param loaded_model: A preloaded Torch model, already on device
    :param loaded_tokenizer: A Torch-compatible tokenizer (HuggingFace usually)
    :param system_prompt: High-level system instruction
    :param user_messages: List of user messages
    :param assistant_messages: List of assistant messages
    :param max_new_tokens: Max tokens to generate
    :param sampler: 'default' => greedy, 'top_p' => nucleus sampling
    :param temperature: For top-p sampling
    :param top_p: For top-p sampling
    :param stop_sequences: If any of these strings appear ANYWHERE in partial text, 
                           we truncate at first occurrence
    :param use_chat_template: If True, attempt to use the built-in chat template
    :return: Final text from the Torch-based model
    """

    # 1) Build final prompt from system/user/assistant messages
    final_prompt = build_chat_prompt(
        tokenizer=loaded_tokenizer,
        system_prompt=system_prompt,
        user_messages=user_messages,
        assistant_messages=assistant_messages,
        add_generation_prompt=True,
        use_template=use_chat_template
    )

    # debug
    logger.debug("torch_chat_generation => final prompt:\n%r", final_prompt)

    # 2) Sampler logic
    if sampler == "default":
        # debug
        logger.debug("torch_chat_generation => greedy_generate_torch")

        # greedy sampler
        output_text = greedy_generate_torch(
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            prompt=final_prompt,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences
        )
    elif sampler == "top_p":
        # debug
        logger.debug("torch_chat_generation => top_p_generate_torch")

        # top p sampler
        output_text = top_p_generate_torch(
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            prompt=final_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences
        )
    else:
        raise ValueError(f"Unsupported sampler: {sampler}")

    logger.debug("torch_chat_generation => raw model output:\n%r", output_text)

    # 3) Remove <|...|> tokens from final text
    cleaned_output = remove_special_tokens_from_text(output_text)
    logger.debug("torch_chat_generation => cleaned output:\n%r", cleaned_output)

    # 4) Return
    return cleaned_output