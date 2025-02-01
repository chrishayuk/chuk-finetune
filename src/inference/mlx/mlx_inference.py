# src/inference/mlx_inference.py
import logging

# mlx imports
try:
    from mlx_lm import load as mlx_load, generate as mlx_generate
except ImportError:
    mlx_load = None
    mlx_generate = None

# imports
from inference.chat_template import build_chat_prompt


def mlx_chat_generation(
    model_name: str,
    system_prompt: str,
    user_messages: list,
    assistant_messages: list,
    max_new_tokens: int
) -> str:
    """
    Minimal Apple MLX-based chat generation function,
    now reusing build_chat_prompt(...) to unify logic.
    """
    if not (mlx_load and mlx_generate):
        raise ImportError("MLX is not installed or cannot be imported.")

    logging.info("Using MLX for inference...")

    # 1) Load model & tokenizer from MLX
    model, tokenizer = mlx_load(model_name)

    # 2) Build the consolidated prompt using the chat template
    prompt_text = build_chat_prompt(
        tokeniser=tokenizer,
        system_prompt=system_prompt,
        user_messages=user_messages,
        assistant_messages=assistant_messages,
        add_generation_prompt=True
    )

    # 3) Generate
    text = mlx_generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt_text,
        max_tokens=max_new_tokens,
        verbose=False
    )

    # strip of text
    return text.strip()
