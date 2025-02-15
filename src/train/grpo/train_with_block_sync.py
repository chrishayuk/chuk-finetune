import logging
import os
from typing import Any, List

# model
from device.torch_device_memory import log_device_memory
from model.checkpoints import save_checkpoint
from model.copy_weights import copy_weights
from model.model_detection import is_torch_model

# train
from train.dataset_loader import get_dataloader

# logger
logger = logging.getLogger(__name__)

def train_grpo_block_sync(
    base_model,
    ref_model,
    tokenizer,
    dataset: List[Any],
    calculate_reward,
    lr: float,
    total_epochs: int = 1,
    batch_size: int = 4,
    G: int = 4,
    device=None,         # e.g. "cpu", "cuda", or "mlx"
    verbose: bool = False,
    kl_coeff: float = 0.1,
    sync_every_n_batches: int = 50,
    shuffle: bool = True,
    # checkpointing (optional)
    checkpoint_dir: str = None,
    checkpoint_every_n_batches: int = None,
    checkpoint_every_n_epochs: int = None
):
    """
    A higher-level training loop that reuses a single trainer instance for multiple batches,
    syncing base_model -> ref_model every `sync_every_n_batches`.
    Also calls a data_loader function properly, avoiding the 'function' object not iterable error.
    """
    total_batches_processed = 0
    final_mean_loss, final_mean_reward = 0.0, 0.0

    # 1) Determine framework & device
    if device == "mlx":
        from train.grpo.mlx.grpo_trainer import GRPOTrainer
        framework = "mlx"
        trainer_device = "mlx"  # or None, depending on how your MLX code handles device
    else:
        from train.grpo.torch.grpo_trainer import GRPOTrainer
        framework = "torch"
        trainer_device = device  # for Torch, might be "cpu", "cuda", etc.

    # Create a single trainer instance for the entire training run.
    from train.optimizer_loader import get_optimizer
    optimizer = get_optimizer(framework, base_model, lr=lr)
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

    # Helper for checkpointing
    def maybe_save_checkpoint(epoch, batch_idx):
        if not checkpoint_dir:
            return
        filename = os.path.join(
            checkpoint_dir,
            f"model_epoch{epoch}_batch{batch_idx}_step{total_batches_processed}.pt"
        )
        logger.info(f"[Checkpoint] Saving model to {filename}")
        save_checkpoint(base_model, filename)

    for epoch in range(1, total_epochs + 1):
        logger.info(f"\n=== Starting Epoch {epoch}/{total_epochs} ===")
        log_device_memory(device=trainer_device, tag=f"Epoch {epoch} Start")

        # 1) Build the data-loader function for this epoch
        data_iter_fn = get_dataloader(framework, dataset, batch_size, shuffle=shuffle)

        # 2) Sync the reference model at the start of the epoch
        logger.info("[Sync] Copy base_model -> ref_model at epoch start.")
        copy_weights(base_model, ref_model, strict=False)
        if is_torch_model(ref_model):
            ref_model.eval()

        log_device_memory(device=trainer_device, tag="After syncing models at epoch start")

        block_batch_count = 0
        batch_idx = 0

        # 3) Iterate over mini-batches
        for batch_items in data_iter_fn():
            batch_idx += 1
            log_device_memory(device=trainer_device, tag=f"Before training batch {batch_idx}")

            # **NEW:** Prepare the batch data before training.
            prepared_data = trainer.prepare_batch_data(batch_items)
            if not prepared_data:
                logger.warning(f"Batch {batch_idx} produced no prepared data; skipping.")
                continue

            # Train on this batch using the preprocessed data
            mean_loss, mean_reward = trainer.train_step(prepared_data)
            log_device_memory(device=trainer_device, tag=f"After training batch {batch_idx}")

            block_batch_count += 1
            total_batches_processed += 1

            # Block sync: refresh the reference model every sync_every_n_batches batches
            if block_batch_count >= sync_every_n_batches:
                logger.info(f"[Sync] Refreshing ref_model after {block_batch_count} batches.")
                copy_weights(base_model, ref_model, strict=True)
                if is_torch_model(ref_model):
                    ref_model.eval()
                block_batch_count = 0
                log_device_memory(device=trainer_device, tag="After block sync")

            # Possibly checkpoint after N mini-batches
            if checkpoint_every_n_batches and (total_batches_processed % checkpoint_every_n_batches == 0):
                maybe_save_checkpoint(epoch, batch_idx)

            logger.info(f"Epoch {epoch}, Batch {batch_idx} => Loss={mean_loss:.4f}, Reward={mean_reward:.4f}")

        # End of epoch
        logger.info(
            f"Epoch {epoch} complete => Last Batch Loss={mean_loss:.4f}, Reward={mean_reward:.4f}"
        )
        log_device_memory(device=trainer_device, tag=f"After epoch {epoch}")

        # Possibly checkpoint after each epoch
        if checkpoint_every_n_epochs and (epoch % checkpoint_every_n_epochs == 0):
            maybe_save_checkpoint(epoch, batch_idx)

    logger.info(
        f"Done training => total of {total_epochs} epochs, {total_batches_processed} mini-batches processed."
    )
    log_device_memory(device=trainer_device, tag="After all training complete")
    return mean_loss, mean_reward