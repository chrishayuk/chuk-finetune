# src/train/optimizer_loader.py

def get_optimizer(framework: str, base_model, lr: float):
    """
    Returns an optimizer for Torch or MLX. 
    The user can extend this with more hyperparameters if needed.
    """
    if framework == "torch":
        # import torch optimizers
        from torch.optim import AdamW

        # Return Torch's AdamW, passing all PyTorch parameters from base_model
        return AdamW(base_model.parameters(), lr=lr)

    elif framework == "mlx":
        # import MLX optimizers
        import mlx.optimizers as optim

        # Return the MLX version of Adam (or AdamW if your MLX version supports it)
        # NOTE: This minimal signature typically doesn't take the parameters;
        # you'll apply updates by calling `optimizer.update(model, grads_dict)` later.
        return optim.Adam(learning_rate=lr)

    else:
        # Unknown framework
        raise ValueError(f"Unknown framework: {framework}")