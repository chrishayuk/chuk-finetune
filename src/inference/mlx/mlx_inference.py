# src/inference/mlx_inference.py
import logging

# imports
from inference.chat_template import build_chat_prompt

try:
    from mlx_lm import generate as mlx_generate
except ImportError:
    mlx_generate = None

def mlx_chat_generation(
    loaded_model,
    loaded_tokenizer,
    system_prompt: str,
    user_messages: list,
    assistant_messages: list,
    max_new_tokens: int
) -> str:
    """
    Apple MLX-based chat generation using a *preloaded* model & tokenizer.

    :param loaded_model: The MLX model object, already loaded (mlx_lm.load).
    :param loaded_tokenizer: The MLX tokenizer object, already loaded.
    :param system_prompt: High-level system instruction.
    :param user_messages: A list of user messages (strings).
    :param assistant_messages: A list of assistant messages (strings).
    :param max_new_tokens: Maximum tokens to generate.
    :return: The generated assistant response (str).
    """
    if mlx_generate is None:
        raise ImportError("MLX is not installed or cannot be imported.")

    logging.info("Using preloaded MLX model for inference...")

    # Build the consolidated prompt using the chat template
    prompt_text = build_chat_prompt(
        tokenizer=loaded_tokenizer,
        system_prompt=system_prompt,
        user_messages=user_messages,
        assistant_messages=assistant_messages,
        add_generation_prompt=True
    )

    # Generate text
    text = mlx_generate(
        model=loaded_model,
        tokenizer=loaded_tokenizer,
        prompt=prompt_text,
        max_tokens=max_new_tokens,
        verbose=True
    )

    # return the text
    return text.strip()
