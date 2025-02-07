# src/model/model_detection.py
def is_mlx_model(model) -> bool:
    """
    Returns True if 'model' appears to be an MLX model by checking for
    `.trainable_parameters()`. Otherwise returns False.
    """
    return hasattr(model, "trainable_parameters") and callable(model.trainable_parameters)

def is_torch_model(model) -> bool:
    """
    Returns True if 'model' is a PyTorch model by checking if it's an instance
    of torch.nn.Module. Otherwise returns False.
    """
    import torch.nn as nn

    # check if pytorch
    return isinstance(model, nn.Module)
