# src/model/torch/torch_checkpoints.py

import logging
import torch

logger = logging.getLogger(__name__)

def save_checkpoint(model, checkpoint_path: str):
    """
    Saves the model's entire state_dict to 'checkpoint_path' via torch.save().
    If you want partial saving (e.g. only adapter params), filter them here.
    """
    # model.state_dict() returns all parameters (trainable + frozen).
    # If you only want trainable, you'd do something like:
    #   {k: v for k,v in model.state_dict().items() if v.requires_grad}
    # But typically, full checkpoint is more common.
    state_dict = model.state_dict()

    # Save to disk
    torch.save(state_dict, checkpoint_path)
    logger.info(f"[Torch] Saved model checkpoint to: {checkpoint_path}")


def load_checkpoint(model, checkpoint_path: str):
    """
    Loads a model's entire state_dict from 'checkpoint_path' into 'model'.
    """
    # Load the saved dict from file
    loaded_sd = torch.load(checkpoint_path, map_location="cpu")

    # Update model in-place
    model.load_state_dict(loaded_sd)
    logger.info(f"[Torch] Loaded model checkpoint from: {checkpoint_path}")
