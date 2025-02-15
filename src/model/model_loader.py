# src/model/model_loader.py
import logging

logger = logging.getLogger(__name__)

def load_model_and_tokenizer(
    model_name_or_path: str,
    device_override: str = None,
    torch_dtype=None
):
    """
    Load a model + tokenizer via MLX or Torch, returning (model, tokenizer, is_mlx).
    If MLX is used, is_mlx=True; otherwise Torch returns is_mlx=None (falsy).
    """

    if device_override == "mlx":
        import mlx.nn as nn
        from mlx_lm import load as mlx_load

        logger.debug("Using MLX. Loading model/tokenizer from %s...", model_name_or_path)
        model, tokenizer = mlx_load(model_name_or_path)

        # Return (model, tokenizer, True) to indicate MLX usage
        return model, tokenizer, True

    else:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        logger.debug("Using Torch. Loading model/tokenizer from %s...", model_name_or_path)

        # Decide a default device if none passed in
        if device_override is None:
            if torch.cuda.is_available():
                device_override = "cuda"
            elif getattr(torch.backends.mps, "is_available", lambda: False)():
                device_override = "mps"
            else:
                device_override = "cpu"

        # Choose default torch_dtype if not provided
        if torch_dtype is None:
            if device_override in ("cuda", "mps"):
                torch_dtype = torch.float16
                #torch_dtype = torch.bfloat16
                #torch_dtype = torch.float32
            else:
                torch_dtype = torch.float32

        logger.debug(f"device_override={device_override}, torch_dtype={torch_dtype}")

        # Use device_map="auto" for Qwen on MPS or multi-GPU
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path,
            torch_dtype=torch_dtype,
            device_map="auto",
            trust_remote_code=True,
        )
        model.gradient_checkpointing_enable()
        model.eval()

        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=True
        )

        # Return (model, tokenizer, None) to indicate Torch usage
        return model, tokenizer, None
