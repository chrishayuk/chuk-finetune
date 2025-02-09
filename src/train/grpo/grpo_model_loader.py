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
        # MLX model
        ref_model.freeze()
    else:
        # Torch model => .eval()
        # You can skip .to(device) if device_map="auto" is used.
        ref_model.eval()

        # If you are NOT using device_map="auto" and have a single device in mind:
        # device = torch.device(device_override)  # or some logic
        # ref_model.to(device)

    # Return everything. If device_map="auto" is used, there's no single device to return.
    return base_model, ref_model, tokenizer, None
