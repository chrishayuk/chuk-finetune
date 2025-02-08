# src/inference/torch/torch_chat_generation.py
import logging

# imports
from inference.mlx.mlx_chat_generation import remove_special_tokens_from_text
from inference.torch.custom_generate_torch import (
    greedy_generate_torch,
    top_p_generate_torch
)

# We'll import the same chat template function you use in MLX
from inference.chat_template import build_chat_prompt

# logging
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
    2) If sampler == 'default', do step-by-step greedy generation.
       If sampler == 'top_p', do step-by-step top-p generation.
    3) Return the final text (possibly truncated if a stop sequence is found).

    This matches the "mlx_chat_generation" style, rather than
    huggingface model.generate streaming logic.

    :param loaded_model: A preloaded Torch model, already on device
    :param loaded_tokenizer: A Torch-compatible tokenizer (HuggingFace usually)
    :param system_prompt: High-level system instruction
    :param user_messages: List of user messages
    :param assistant_messages: List of assistant messages
    :param max_new_tokens: Max tokens to generate
    :param sampler: 'default' => greedy, 'top_p' => nucleus sampling
    :param temperature: For top-p sampling
    :param top_p: For top-p sampling
    :param stop_sequences: If any of these strings appear ANYWHERE in partial text, we truncate
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

    logger.debug("torch_chat_generation => final prompt:\n%r", final_prompt)

    # 2) Sampler logic
    if sampler == "default":
        logger.debug("torch_chat_generation => greedy_generate_torch")
        output_text = greedy_generate_torch(
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            prompt=final_prompt,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences  # We'll add 'stop_sequences' support below
        )
    elif sampler == "top_p":
        logger.debug("torch_chat_generation => top_p_generate_torch")
        output_text = top_p_generate_torch(
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            prompt=final_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences  # We'll add 'stop_sequences' support
        )
    else:
        raise ValueError(f"Unsupported sampler: {sampler}")
    
    # debug
    logger.debug("torch_chat_generation => raw model output:\n%r", output_text)

    # 3) Remove <|...|> tokens from final text
    cleaned_output = remove_special_tokens_from_text(output_text)
    logger.debug("torch_chat_generation => cleaned output:\n%r", cleaned_output)

    # 4) Return
    return cleaned_output
