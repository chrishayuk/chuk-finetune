# src/mlx_model_utils.py

import os
from pathlib import Path
from typing import Optional, Dict, Any

import mlx.nn as nn
import mlx.core as mx

try:
    from huggingface_hub import snapshot_download
except ImportError:
    snapshot_download = None


def get_model_path(path_or_hf_repo: str) -> Path:
    """
    Check if `path_or_hf_repo` exists locally; if not, attempt to snapshot_download
    from Hugging Face. Return the local Path.

    Raises FileNotFoundError if neither a local path nor HF download can be done.
    """
    model_path = Path(path_or_hf_repo)
    if model_path.exists():
        # It's a local directory
        return model_path

    # If huggingface_hub is available, try downloading from HF
    if snapshot_download is None:
        raise FileNotFoundError(
            f"MLX: The path {path_or_hf_repo} does not exist locally, "
            f"and huggingface_hub is not installed to download it."
        )

    print(f"[MLX] Local path {path_or_hf_repo} not found. Attempting HF download...")
    try:
        cache_dir = snapshot_download(repo_id=path_or_hf_repo)
        model_path = Path(cache_dir)
        print(f"[MLX] Downloaded HF repo to: {model_path}")
        return model_path
    except Exception as e:
        raise FileNotFoundError(
            f"[MLX] Could not find or download model '{path_or_hf_repo}'. "
            "Verify spelling or HF permissions."
        ) from e


def load_model(model_path: Path, lazy: bool = False):
    """
    Placeholder for a real MLX model load. In production:
      - parse config.json
      - locate *.safetensors
      - construct an MLX model
      - load weights
    """
    print(f"[MLX] Loading MLX model from local path: {model_path}")
    # Minimal demonstration: return an empty nn.Module and dummy config
    model = nn.Module()
    model.eval()
    config = {}
    # If you need 'lazy' logic, you can skip evaluating all parameters now
    return model, config


def load_tokenizer(
    model_path: Path,
    tokenizer_config: Dict[str, Any] = None
):
    """
    Placeholder for MLX tokenizer loading logic.
    E.g. reading a tokenizer.json, or using HF's tokenizer wrapped in an MLX class.
    """
    print(f"[MLX] Loading tokenizer from local path: {model_path}")
    return "dummy_mlx_tokenizer"


def load_adapters(model: nn.Module, adapter_path: str) -> nn.Module:
    """
    Minimal stand-in for applying LoRA or other adapters to an MLX model.
    """
    if adapter_path is not None:
        print(f"[MLX] Loading adapters from {adapter_path}")
    # In a real script, you'd load adapter weights from that path
    return model


def load_mlx_model_and_tokenizer(
    model_name_or_path: str,
    adapter_path: Optional[str] = None,
    lazy: bool = False,
    tokenizer_config: Optional[Dict[str, Any]] = None
):
    """
    Load an MLX model and tokenizer using the logic from Apple MLX's loader scripts.
    Returns (model, tokenizer, None) because MLX doesn't have the Torch device concept.

    :param model_name_or_path: local directory or HF repo name
    :param adapter_path: optional path for LoRA adapter weights
    :param lazy: if True, don't fully load model into memory yet
    :param tokenizer_config: extra settings for tokenizer
    :return: (nn.Module, tokenizer_object, device=None)
    """
    if tokenizer_config is None:
        tokenizer_config = {}

    # 1) Ensure the model is locally available (or download from HF)
    model_path = get_model_path(model_name_or_path)

    # 2) Load the model & config
    model, config = load_model(model_path, lazy=lazy)

    # 3) Apply adapters (LoRA, etc.) if needed
    if adapter_path is not None:
        model = load_adapters(model, adapter_path)

    # 4) Load tokenizer
    tokenizer = load_tokenizer(model_path, tokenizer_config=tokenizer_config)

    # Return a triple. MLX doesn't rely on the Torch device concept.
    return model, tokenizer, None
