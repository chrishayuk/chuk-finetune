# src/inference/mlx_inference.py
import logging

try:
    from mlx_lm import load as mlx_load, generate as mlx_generate
except ImportError:
    mlx_load = None
    mlx_generate = None

def mlx_chat_generation(model_name: str,
                        system_prompt: str,
                        user_messages: list,
                        max_new_tokens: int) -> str:
    """
    Minimal Apple MLX-based chat generation function.
    No streaming or advanced stats here by default.
    """
    if not (mlx_load and mlx_generate):
        raise ImportError("MLX is not installed or cannot be imported.")

    logging.info("Using MLX for inference...")

    # 1) Load model & tokenizer from MLX
    model, tokenizer = mlx_load(model_name)

    # 2) Build a minimal chat prompt
    messages = [{"role": "system", "content": system_prompt}]
    if user_messages:
        messages.append({"role": "user", "content": user_messages[-1]})

    # Convert to a prompt via MLX’s chat template
    prompt_text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
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
    return text.strip()
