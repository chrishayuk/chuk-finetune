# src/train/grpo_trainer.py
from train.optimizer_loader import get_optimizer
from train.get_dataloader import get_dataloader
from train.generic_train import train as generic_train

def train_grpo(
    base_model,
    ref_model,
    tokenizer,
    dataset,            # list or dataset object for either Torch or MLX
    calculate_reward,   # function returning (score, feedback_text)
    lr: float,
    epochs: int = 1,
    batch_size: int = 4,
    G: int = 4,
    device=None,        # "mlx", "torch", or possibly None
    verbose: bool = False,
    kl_coeff: float = 0.1
):
    """
    Unified training loop that decides if we're using MLX or Torch, 
    then instantiates the correct GRPOTrainer and runs generic_train.
    """
    # 1) Determine the framework string
    if isinstance(device, str):
        dev_str = device.strip().lower()
    else:
        # if device is None or bool, decide a default
        if device is True:
            dev_str = "mlx"
        else:
            dev_str = "torch"  # or None => default to "torch"? It's up to you

    # 2) Dynamically import the matching GRPOTrainer
    if dev_str == "mlx":
        from train.grpo.mlx.grpo_trainer import GRPOTrainer
        framework = "mlx"
    else:
        from train.grpo.torch.grpo_trainer import GRPOTrainer
        framework = "torch"

    # get the optmizer (will handle torch or mlx)
    optimizer = get_optimizer(framework, base_model, lr=lr)

    # get the dataloader (will handle torch or mlx)
    data_iterator_fn = get_dataloader(framework, dataset, batch_size, shuffle=True)

    # create the trainer object
    trainer = GRPOTrainer(
        model=base_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        calculate_reward=calculate_reward,
        G=G,
        kl_coeff=kl_coeff,
        device=dev_str,
        verbose=verbose
    )

    # call the generic training loop
    mean_loss, mean_reward = generic_train(
        trainer=trainer,
        data_iterator=data_iterator_fn,
        epochs=epochs
    )

    # 6) Return aggregated metrics
    return mean_loss, mean_reward
