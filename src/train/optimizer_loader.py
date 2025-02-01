# src/train/optimizer_loader.py
def get_optimizer(framework: str, base_model, lr: float):
    """
    Returns an optimizer for Torch or MLX. 
    The user can extend this with more hyperparameters if needed.
    """
    if framework == "torch":
        from torch.optim import AdamW
        return AdamW(base_model.parameters(), lr=lr)

    elif framework == "mlx":
        import mlx.optimizers as optim
        return optim.Adam(learning_rate=lr)

    else:
        raise ValueError(f"Unknown framework: {framework}")
