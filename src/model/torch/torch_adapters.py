# src/mode/torch/torch_adapters.py
import torch
import logging

# logger
logger = logging.getLogger(__name__)

def save_adapters(model, adapter_path: str):
    """
    Saves all parameters that have requires_grad=True from a PyTorch model
    to a .pt file via torch.save().
    """
    # Collect only trainable parameters
    adapter_parameters = {
        name: param.detach().cpu()  # Move to CPU for saving
        for name, param in model.named_parameters()
        if param.requires_grad
    }
    
    # Save to disk with torch.save
    torch.save(adapter_parameters, adapter_path)
    logger.info(f"Saved adapter parameters to: {adapter_path}")


def load_adapters(model, adapter_path: str):
    """
    Loads adapter parameters from a .pt file into a PyTorch model.
    Only updates parameters that have requires_grad=True.
    """
    # Load the saved dict of adapter params
    loaded_adapters = torch.load(adapter_path, map_location="cpu")

    # Get the model’s entire state_dict
    model_dict = model.state_dict()

    # Verify that each loaded param is in the model
    for param_key in loaded_adapters.keys():
        if param_key not in model_dict:
            raise KeyError(f"Model has no parameter named '{param_key}'.")

    # Update the model dict with the adapter weights
    model_dict.update(loaded_adapters)

    # Load the updated state_dict back into the model
    model.load_state_dict(model_dict)
    logger.info(f"Loaded adapter parameters from: {adapter_path}")
