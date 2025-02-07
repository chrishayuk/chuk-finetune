# src/model/mlx/mlx_adapters.py
import logging
import mlx.core as mx
from mlx.utils import tree_flatten

# get the loggers
logger = logging.getLogger(__name__)

def save_adapters(model, adapter_path: str):
    """
    Saves the trainable adapter parameters to a .safetensors file.
    """
    # Convert the model’s trainable parameters into a dictionary
    adapter_weights = dict(tree_flatten(model.trainable_parameters()))

    # Save to disk
    mx.save_safetensors(adapter_path, adapter_weights)

    # Log the result
    logger.info("Saved adapters to: %s", adapter_path)


def load_adapters(model, adapter_path: str):
    """
    Loads the adapter parameters from a .safetensors file into the model.
    """
    # Read the saved dictionary of adapter weights
    loaded_weights = mx.load_safetensors(adapter_path)
    
    # Flatten the model params so we can load one-by-one
    model_params = dict(tree_flatten(model.trainable_parameters()))

    # Each key in 'loaded_weights' should match a key in 'model_params'.
    # Update the model's parameters in-place.
    for param_key, param_array in loaded_weights.items():
        if param_key not in model_params:
            raise KeyError(f"Model has no parameter named '{param_key}'.")
        model_params[param_key][...] = param_array

    # Log the result
    logger.info("Loaded adapters from: %s", adapter_path)
