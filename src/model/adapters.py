# src/model/adapters.py
import logging
import os

# imports local modules
from model.model_detection import is_mlx_model, is_torch_model
from model.mlx.mlx_adapters import save_adapters as mlx_save_adapters, load_adapters as mlx_load_adapters
from model.torch.torch_adapters import save_adapters as torch_save_adapters, load_adapters as torch_load_adapters

# logger
logger = logging.getLogger(__name__)

def save_adapters(model, adapter_path: str):
    """
    Saves adapter weights, auto-detecting whether 'model' is MLX or PyTorch.
    
    :param model: An MLX model (with .trainable_parameters()) or a PyTorch model (nn.Module).
    :param adapter_path: The path (str) where the adapter weights will be saved.
    """

    # check if mlx
    if is_mlx_model(model):
        # save adapters as mlx
        mlx_save_adapters(model, adapter_path)
    elif is_torch_model(model):
        # save adapters as torch
        torch_save_adapters(model, adapter_path)
    else:
        raise TypeError(
            "save_adapters: Could not detect if model is MLX or PyTorch. "
            "Please ensure your model is one of these types."
        )


def load_adapters(model, adapter_path: str):
    """
    Loads adapter weights, auto-detecting whether 'model' is MLX or PyTorch.

    :param model: An MLX model (with .trainable_parameters()) or a PyTorch model (nn.Module).
    :param adapter_path: The path (str) where the adapter weights are located.
    :raises FileNotFoundError: If the specified adapter_path does not exist.
    """
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter file not found: {adapter_path}")

    # check if mlx
    if is_mlx_model(model):
        # save adapters as mlx
        mlx_load_adapters(model, adapter_path)
    elif is_torch_model(model):
        # save adapters as torch
        torch_load_adapters(model, adapter_path)
    else:
        raise TypeError(
            "load_adapters: Could not detect if model is MLX or PyTorch. "
            "Please ensure your model is one of these types."
        )
