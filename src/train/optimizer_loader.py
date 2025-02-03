# src/train/optimizer_loader.py
def get_optimizer(framework: str, base_model, lr: float):
    """
    Returns an optimizer for Torch or MLX. 
    The user can extend this with more hyperparameters if needed.
    """
    if framework == "torch":
        # import torch optimizers
        from torch.optim import AdamW

        # return the torch version of adamw
        return AdamW(base_model.parameters(), lr=lr)

    elif framework == "mlx":
        # import mlx optimizers
        import mlx.optimizers as optim

        # return the mlx version of adamw
        return optim.Adam(learning_rate=lr)

    else:
        # unknow framework
        raise ValueError(f"Unknown framework: {framework}")
