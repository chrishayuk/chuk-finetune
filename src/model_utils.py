# src/model_utils.py
def load_model_and_tokenizer(
    model_name_or_path: str,
    device_override: str = None
):
    """
    Load a model + tokenizer using either Torch or MLX, depending on device_override.

    If device_override == "mlx", use Apple MLX logic.
    Otherwise, assume Torch for "cpu", "cuda", "mps", or None.
    """
    # If device_override is "mlx", do MLX
    if device_override == "mlx":
        # ------------------ MLX Path ------------------
        import mlx.nn as nn
        from mlx_lm import load as mlx_load

        print(f"[INFO] Using MLX. Loading model/tokenizer from {model_name_or_path}...")
        model, tokenizer = mlx_load(model_name_or_path)

        # MLX manages device differently, so we return None
        device = None

        # return model, tokenizer and device
        return model, tokenizer, device

    else:
        # ------------------ Torch Path ------------------
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from torch_device_selection import DeviceSelector

        print(f"[INFO] Using Torch. Loading model/tokenizer from {model_name_or_path}...")

        # Determine actual device if not passed
        if device_override is None:
            # auto-detect (CUDA if available, else CPU)
            device_override = "cuda" if torch.cuda.is_available() else "cpu"

        # set the device
        device = torch.device(device_override)

        # Load model
        model = AutoModelForCausalLM.from_pretrained(
            model_name_or_path, torch_dtype="auto"
        )

        # move the model to the device
        model.to(device)

        # put in eval mode
        model.eval()

        # Load tokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name_or_path)

        # return the model, tokenizer, device
        return model, tokenizer, device
