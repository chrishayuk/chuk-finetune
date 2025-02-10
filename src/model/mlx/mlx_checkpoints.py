# src/model/mlx/mlx_checkpoints.py

import logging
import mlx.core as mx
from mlx.utils import tree_flatten

logger = logging.getLogger(__name__)

def save_checkpoint(model, checkpoint_path: str):
    """
    Saves the entire model's trainable parameters (or all parameters, depending on your preference)
    to a .safetensors or .json file, etc., for MLX.
    """
    # Convert the model’s parameters into a flat dictionary
    # You could choose 'model.trainable_parameters()' or 'model.parameters()'
    all_params = dict(tree_flatten(model.trainable_parameters()))

    # Example: save via safetensors approach
    mx.save_safetensors(checkpoint_path, all_params)

    logger.info(f"[MLX] Saved model checkpoint to: {checkpoint_path}")


def load_checkpoint(model, checkpoint_path: str):
    """
    Loads the model’s trainable parameters from a file into the MLX model.
    """
    # Load dictionary from file
    loaded_params = mx.load_safetensors(checkpoint_path)
    
    # Flatten model params so we can copy each
    model_params = dict(tree_flatten(model.trainable_parameters()))

    # For each param, overwrite the data from loaded_params
    for param_key, param_array in loaded_params.items():
        if param_key not in model_params:
            raise KeyError(f"MLX model has no parameter named '{param_key}'.")
        model_params[param_key][...] = param_array

    logger.info(f"[MLX] Loaded model checkpoint from: {checkpoint_path}")
