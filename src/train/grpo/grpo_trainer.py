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
    Unified training loop for GRPO:
      1. Determines the framework ("mlx" or "torch") based on `device`.
      2. Dynamically imports the corresponding GRPOTrainer.
      3. Instantiates an optimizer (MLX or torch).
      4. Retrieves a data iterator that yields batches.
      5. Creates the GRPOTrainer instance with the given models and params.
      6. Delegates the actual training loop to `generic_train`.
      7. Returns aggregated (mean_loss, mean_reward).
    """

    # 1) Determine the framework string
    if isinstance(device, str):
        dev_str = device.strip().lower()
    else:
        # If device is None or boolean, pick a default
        if device is True:
            dev_str = "mlx"
        else:
            dev_str = "torch"

    # 2) Dynamically import the matching GRPOTrainer
    if dev_str == "mlx":
        from train.grpo.mlx.grpo_trainer import GRPOTrainer
        framework = "mlx"
    else:
        from train.grpo.torch.grpo_trainer import GRPOTrainer
        framework = "torch"

    # 3) Get the optimizer (handles torch or MLX under the hood)
    optimizer = get_optimizer(framework, base_model, lr=lr)

    # 4) Get the data iterator (handles torch or MLX under the hood)
    #    Returns a function that can be called as data_iterator_fn(batch_size)
    data_iterator_fn = get_dataloader(framework, dataset, batch_size, shuffle=True)

    # 5) Create the GRPO trainer object
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

    # 6) Call the generic training loop
    mean_loss, mean_reward = generic_train(
        trainer=trainer,
        data_iterator=data_iterator_fn,
        epochs=epochs,
        batch_size=batch_size,
        log_per_batch=verbose
    )

    # 7) Return the aggregated metrics
    return mean_loss, mean_reward
