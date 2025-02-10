# src/model/checkpoints.py

import logging
import os

# model detection
from model.model_detection import is_mlx_model, is_torch_model

# logging
logger = logging.getLogger(__name__)

def save_checkpoint(model, checkpoint_path: str):
    """
    Auto-detect whether 'model' is MLX or Torch, then save a checkpoint to 'checkpoint_path'.

    :param model:
        - An MLX model (with .trainable_parameters()),
        - A PyTorch model (nn.Module).
    :param checkpoint_path: The file or directory path where you want to save the checkpoint.
    """
    # Ensure output directory exists
    dirpath = os.path.dirname(checkpoint_path)
    
    if dirpath and not os.path.exists(dirpath):
        os.makedirs(dirpath, exist_ok=True)

    if is_mlx_model(model):
        logger.info(f"Saving MLX model checkpoint to {checkpoint_path}")
        # Import only if we detect MLX
        from model.mlx.mlx_checkpoints import save_checkpoint as mlx_save_checkpoint
        mlx_save_checkpoint(model, checkpoint_path)

    elif is_torch_model(model):
        logger.info(f"Saving Torch model checkpoint to {checkpoint_path}")
        # Import only if we detect PyTorch
        from model.torch.torch_checkpoints import save_checkpoint as torch_save_checkpoint
        torch_save_checkpoint(model, checkpoint_path)

    else:
        raise TypeError(
            "save_checkpoint: Could not detect if model is MLX or PyTorch. "
            "Please ensure your model is one of these types."
        )


def load_checkpoint(model, checkpoint_path: str):
    """
    Auto-detect whether 'model' is MLX or Torch, then load a checkpoint from 'checkpoint_path'.

    :param model:
        - An MLX model (with .trainable_parameters()),
        - A PyTorch model (nn.Module).
    :param checkpoint_path: The file or directory path from which you want to load the checkpoint.
    :raises FileNotFoundError: If the specified checkpoint_path does not exist.
    """
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if is_mlx_model(model):
        logger.info(f"Loading MLX model checkpoint from {checkpoint_path}")
        # Import only if we detect MLX
        from model.mlx.mlx_checkpoints import load_checkpoint as mlx_load_checkpoint
        mlx_load_checkpoint(model, checkpoint_path)

    elif is_torch_model(model):
        logger.info(f"Loading Torch model checkpoint from {checkpoint_path}")
        # Import only if we detect PyTorch
        from model.torch.torch_checkpoints import load_checkpoint as torch_load_checkpoint
        torch_load_checkpoint(model, checkpoint_path)

    else:
        raise TypeError(
            "load_checkpoint: Could not detect if model is MLX or PyTorch. "
            "Please ensure your model is one of these types."
        )
