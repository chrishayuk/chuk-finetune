# src/train/unified_grpo_trainer.py
from train.optimizer_loader import get_optimizer
from train.get_dataloader import get_dataloader

def train_grpo(
    base_model,
    ref_model,
    tokenizer,
    dataset,            # e.g. for Torch: DataLoader or list of dict
    calculate_reward,   # function returning float
    lr: float,
    epochs: int = 1,
    batch_size: int = 4,
    G: int = 4,
    device: str = None,
    verbose: bool = False
):
    """
    Unified training loop for MLX and Torch.
    Ensures that dataset batches are dicts, not stringified versions.
    """
    dev_str = (device.strip().lower() if device else None)

    if dev_str == "mlx":
        # MLX branch
        from train.mlx.grpo_trainer import train_grpo as mlx_train_grpo
        optimizer = get_optimizer("mlx", base_model, lr=lr)

        # get_dataloader returns a function, so we store it as data_iterator_fn
        data_iterator_fn = get_dataloader("mlx", dataset, batch_size, shuffle=True)

        # call mlx
        return mlx_train_grpo(
            base_model=base_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            data_iterator=data_iterator_fn,
            calculate_reward=calculate_reward,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            G=G,
            device=device,
            verbose=verbose
        )

    else:
        # Torch branch
        from train.torch.grpo_trainer import train_grpo as torch_train_grpo
        optimizer = get_optimizer("torch", base_model, lr=lr)

        # get_dataloader returns a function, so we store it as data_loader_fn
        data_loader_fn = get_dataloader("torch", dataset, batch_size, shuffle=True)

        # call torch
        return torch_train_grpo(
            base_model=base_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            data_loader=data_loader_fn,
            calculate_reward=calculate_reward,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            G=G,
            device=device,
            verbose=verbose
        )
