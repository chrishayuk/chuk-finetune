# src/model/adapters.py
import logging
import os

# imports
from model.model_detection import is_mlx_model, is_torch_model

# logging
logger = logging.getLogger(__name__)

def save_adapters(model, adapter_path: str):
    """
    Saves adapter weights, auto-detecting whether 'model' is MLX or PyTorch.
    
    :param model: 
        An MLX model (with .trainable_parameters()) or a PyTorch model (nn.Module).
    :param adapter_path: 
        The path (str) where the adapter weights will be saved.
    """

    # Ensure directory if needed
    dirpath = os.path.dirname(adapter_path)
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    if is_mlx_model(model):
        logger.info(f"Saving MLX adapters to: {adapter_path}")
        # Lazy import only if we detect MLX
        from model.mlx.mlx_adapters import save_adapters as mlx_save_adapters
        mlx_save_adapters(model, adapter_path)

    elif is_torch_model(model):
        logger.info(f"Saving Torch adapters to: {adapter_path}")
        # Lazy import only if we detect Torch
        from model.torch.torch_adapters import save_adapters as torch_save_adapters
        torch_save_adapters(model, adapter_path)

    else:
        raise TypeError(
            "save_adapters: Could not detect if model is MLX or PyTorch. "
            "Please ensure your model is one of these types."
        )


def load_adapters(model, adapter_path: str):
    """
    Loads adapter weights, auto-detecting whether 'model' is MLX or PyTorch.

    :param model: 
        An MLX model (with .trainable_parameters()) or a PyTorch model (nn.Module).
    :param adapter_path: 
        The path (str) where the adapter weights are located.
    :raises FileNotFoundError: If the specified adapter_path does not exist.
    """
    if not os.path.exists(adapter_path):
        raise FileNotFoundError(f"Adapter file not found: {adapter_path}")

    if is_mlx_model(model):
        logger.info(f"Loading MLX adapters from: {adapter_path}")
        from model.mlx.mlx_adapters import load_adapters as mlx_load_adapters
        mlx_load_adapters(model, adapter_path)

    elif is_torch_model(model):
        logger.info(f"Loading Torch adapters from: {adapter_path}")
        from model.torch.torch_adapters import load_adapters as torch_load_adapters
        torch_load_adapters(model, adapter_path)

    else:
        raise TypeError(
            "load_adapters: Could not detect if model is MLX or PyTorch. "
            "Please ensure your model is one of these types."
        )
