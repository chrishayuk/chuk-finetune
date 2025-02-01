# src/train/unified_grpo_trainer.py
from train.optimizer_loader import get_optimizer
from train.dataset_loader import get_dataloader

def train_grpo(
    base_model,
    ref_model,
    tokenizer,
    verifier,
    dataset,            # Torch => e.g. dataset/DataLoader
    calculate_reward,   # function returning float
    lr: float,
    epochs: int = 1,
    batch_size: int = 4,
    G: int = 4,
    device: str = None,
    verbose: bool = False
):
    dev_str = (device.strip().lower() if device else None)

    if dev_str == "mlx":
        # MLX branch
        from train.mlx.grpo_trainer import train_grpo as mlx_train_grpo
        optimizer = get_optimizer("mlx", base_model, lr=lr)
        data_iterator = get_dataloader("mlx", dataset, batch_size, shuffle=True)

        return mlx_train_grpo(
            base_model=base_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            verifier=verifier,
            data_iterator=data_iterator,
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
        data_loader = get_dataloader("torch", dataset, batch_size, shuffle=True)

        return torch_train_grpo(
            base_model=base_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            verifier=verifier,
            data_loader=data_loader,
            calculate_reward=calculate_reward,
            optimizer=optimizer,
            epochs=epochs,
            batch_size=batch_size,
            G=G,
            device=device,
            verbose=verbose
        )
