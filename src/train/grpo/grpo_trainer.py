# src/train/grpo_trainer.py

from train.optimizer_loader import get_optimizer
from train.get_dataloader import get_dataloader

# Import the single generator-based generic_train
from train.generic_train import generic_train

def train_grpo(
    base_model,
    ref_model,
    tokenizer,
    dataset,
    calculate_reward,
    lr: float,
    epochs: int = 1,
    batch_size: int = 4,
    G: int = 4,
    device=None,
    verbose: bool = False,
    kl_coeff: float = 0.1,
    as_generator: bool = False
):
    """
    A single GRPO entry point. It always uses the generator-based `generic_train`.

    If as_generator=False, we fully consume the generator inside this function
    and return (mean_loss, mean_reward).

    If as_generator=True, we return the generator itself so the caller
    can iterate over it for step-by-step events.

    Steps:
      1) Determine framework (mlx or torch).
      2) Import GRPOTrainer dynamically.
      3) Instantiate optimizer & data iterator.
      4) Build the trainer (GRPOTrainer).
      5) Call generic_train(...) => yields events.
      6) If as_generator=False => consume the yields here, returning final results.
         If as_generator=True => return the generator to the caller.
    """

    # 1) Determine framework
    if isinstance(device, str):
        dev_str = device.strip().lower()
        if dev_str not in ["mlx", "torch"]:
            dev_str = "torch"
    else:
        dev_str = "mlx" if device is True else "torch"

    # 2) Dynamic import
    if dev_str == "mlx":
        from train.grpo.mlx.grpo_trainer import GRPOTrainer
        framework = "mlx"
    else:
        from train.grpo.torch.grpo_trainer import GRPOTrainer
        framework = "torch"

    # create the optimizer (framework independent)
    optimizer = get_optimizer(framework, base_model, lr=lr)

    # create the data iterator (franework independent)
    data_iterator_fn = get_dataloader(framework, dataset, batch_size, shuffle=True)

    # create the trainer (framework independent)
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

    # call the generator-based training loop
    gen = generic_train(
        trainer=trainer,
        data_iterator=data_iterator_fn,
        epochs=epochs,
        batch_size=batch_size
    )

    # cecide how to expose the generator
    if as_generator:
        # Return the generator directly
        return gen
    else:
        # We'll consume it here, find the final "train_end" event, and return
        final_mean_loss = 0.0
        final_mean_reward = 0.0

        for event in gen:
            if event.get("train_end"):
                final_mean_loss = event["mean_loss"]
                final_mean_reward = event["mean_reward"]
                break  # Done; exit loop

        # return the final mean loss and reward
        return final_mean_loss, final_mean_reward
