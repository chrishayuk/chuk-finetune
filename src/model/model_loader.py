# src/model/model_loader.py
import logging
logger = logging.getLogger(__name__)

def load_model_and_tokenizer(model_name_or_path: str, device_override: str = None):
    """
    Load a model and tokenizer using either Torch or MLX, depending on device_override.

    If device_override == "mlx", use Apple MLX logic and return a flag indicating MLX usage.
    Otherwise, assume Torch for "cpu", "cuda", "mps", or None and return False.
    """
    if device_override == "mlx":
        # ------------------ MLX Path ------------------
        import mlx.nn as nn
        from mlx_lm import load as mlx_load

        # load the model
        logger.info("Using MLX. Loading model/tokenizer from %s...", model_name_or_path)
        model, tokenizer = mlx_load(model_name_or_path)

        # MLX manages devices differently, so we return a flag indicating MLX is in use.
        return model, tokenizer, True

    else:
        # ------------------ Torch Path ------------------
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.info("Using Torch. Loading model/tokenizer from %s...", model_name_or_path)

        # Determine the device if not passed
        if device_override is None:
            # Auto-detect: use CUDA if available, else CPU
            device_override = "cuda" if torch.cuda.is_available() else "cpu"

        # get the torch device
        device = torch.device(device_override)

        # Load the model and move it to the correct device
        model = AutoModelForCausalLM.from_pretrained(model_name_or_path, torch_dtype="auto")
        model.to(device)
        model.eval()

        # Load the tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # return the model, tokenizer
        return model, tokenizer, False
