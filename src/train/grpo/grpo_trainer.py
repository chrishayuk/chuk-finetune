# src/train/grpo_trainer.py
from train.optimizer_loader import get_optimizer
from train.get_dataloader import get_dataloader
from train.generic_train import generic_train

# imports
from train.trainer_events import TrainerEvent

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
    device=None,        # might be "mps", "cuda", "cpu", or "mlx"
    verbose: bool = False,
    kl_coeff: float = 0.1,
    as_generator: bool = False
):
    """
    A single GRPO entry point, uses the generator-based `generic_train`.

    If as_generator=False, we fully consume the generator inside this function
    and return (mean_loss, mean_reward).

    If as_generator=True, we return the generator itself so the caller
    can iterate over it for step-by-step events.
    """

    # 1) Determine framework & device
    if device == "mlx":
        from train.grpo.mlx.grpo_trainer import GRPOTrainer
        framework = "mlx"
        trainer_device = True  # or None, depending on how your MLX code handles device
    else:
        from train.grpo.torch.grpo_trainer import GRPOTrainer
        framework = "torch"
        trainer_device = device  # for Torch, might be "cpu", "cuda", etc.

    # 2) Create the optimizer (framework independent)
    optimizer = get_optimizer(framework, base_model, lr=lr)

    # 3) Create the data iterator (framework independent)
    data_iterator_fn = get_dataloader(framework, dataset, batch_size, shuffle=True)

    # 4) Build the trainer
    trainer = GRPOTrainer(
        model=base_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        calculate_reward=calculate_reward,
        G=G,
        kl_coeff=kl_coeff,
        device=trainer_device,
        verbose=verbose
    )

    # 5) Call generic_train(...) => yields TrainerEvent objects
    gen = generic_train(
        trainer=trainer,
        data_iterator=data_iterator_fn,
        epochs=epochs,
        batch_size=batch_size
    )

    # 6) Decide how to expose the generator
    if as_generator:
        # return the generator, so the user can do:
        #   for event in train_grpo(...):
        #       ...
        return gen
    else:
        final_mean_loss = 0.0
        final_mean_reward = 0.0

        # consume events
        for event in gen:
            # event is a TrainerEvent (pydantic or dataclass)
            if event.event_type == "train_end":
                final_mean_loss = event.mean_loss or 0.0
                final_mean_reward = event.mean_reward or 0.0
                break

        return final_mean_loss, final_mean_reward
