# src/train/grpo_model_loader.py

from cli.train.logger_config import logger, color_text, BOLD
from model.model_loader import load_model_and_tokenizer

def load_models(model_name, device_override):
    logger.info(color_text(f"Loading base model & tokenizer: {model_name}", BOLD))

    # 1) Load the base model & tokenizer
    base_model, tokenizer, is_mlx_base = load_model_and_tokenizer(
        model_name_or_path=model_name,
        device_override=device_override,
    )

    # 2) Load the reference model & tokenizer
    logger.info("Loading reference model (KL/PPO) ...")
    ref_model, _, is_mlx_ref = load_model_and_tokenizer(
        model_name_or_path=model_name,
        device_override=device_override,
    )

    # 3) If reference model is MLX, freeze it; else (Torch), just do .eval().
    if is_mlx_ref:
        ref_model.freeze()
    else:
        ref_model.eval()

    # 4) Decide what 'device' to return
    #    If either the base or ref model is MLX, we return "mlx". 
    #    Otherwise, you can return the 'device_override' or a default like "cpu".
    if is_mlx_base or is_mlx_ref:
        final_device = "mlx"
    else:
        # If you wanted a fallback for Torch, 
        # you might do device_override or "cpu" or "cuda" if you want.
        final_device = device_override or "cpu"

    # Return the final device instead of None
    return base_model, ref_model, tokenizer, final_device