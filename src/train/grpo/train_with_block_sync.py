# src/train/grpo/train_with_block_sync.py

import logging
import os
from typing import Any, List
from train.dataset_loader import get_dataloader
from train.grpo.grpo_trainer import train_grpo
from model.checkpoints import save_checkpoint

# NEW IMPORTS:
from model.copy_weights import copy_weights
from model.model_detection import is_torch_model

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
    A higher-level training loop that calls `train_grpo(...)` repeatedly, 
    syncing base_model -> ref_model every `sync_every_n_batches`.
    Also calls a data_loader function properly (we call the returned function),
    avoiding the 'function' object not iterable error.
    """
    total_batches_processed = 0
    final_mean_loss, final_mean_reward = 0.0, 0.0

    # Helper if you do checkpointing
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

        # 1) Build the data-loader function
        data_iter_fn = get_dataloader("torch", dataset, batch_size, shuffle=shuffle)

        # 2) Sync the reference model at the start of the epoch
        logger.info("[Sync] Copy base_model -> ref_model at epoch start.")
        copy_weights(base_model, ref_model, strict=False)
        if is_torch_model(ref_model):
            ref_model.eval()

        block_batch_count = 0
        batch_idx = 0

        # 3) Actually call data_iter_fn(...) with or without a batch_size override
        for batch_items in data_iter_fn():
            batch_idx += 1

            # 3.1) Train on this single batch => "epochs=1"
            mean_loss, mean_reward = train_grpo(
                base_model=base_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                dataset=batch_items,
                calculate_reward=calculate_reward,
                lr=lr,
                epochs=1,            # one pass for this batch
                batch_size=batch_size,
                G=G,
                device=device,
                verbose=verbose,
                kl_coeff=kl_coeff,
                as_generator=False
            )

            block_batch_count += 1
            total_batches_processed += 1

            # Block sync
            if block_batch_count >= sync_every_n_batches:
                logger.info(f"[Sync] Refreshing ref_model after {block_batch_count} batches.")
                copy_weights(base_model, ref_model, strict=True)
                if is_torch_model(ref_model):
                    ref_model.eval()
                block_batch_count = 0

            # Possibly checkpoint after N mini-batches
            if checkpoint_every_n_batches and (total_batches_processed % checkpoint_every_n_batches == 0):
                maybe_save_checkpoint(epoch, batch_idx)

            logger.info(f"Epoch {epoch}, Batch {batch_idx} => Loss={mean_loss:.4f}, Reward={mean_reward:.4f}")

        # End of epoch
        logger.info(
            f"Epoch {epoch} complete => last batch Loss={mean_loss:.4f}, Reward={mean_reward:.4f}"
        )

        # Possibly checkpoint after each epoch
        if checkpoint_every_n_epochs and (epoch % checkpoint_every_n_epochs == 0):
            maybe_save_checkpoint(epoch, batch_idx)

    logger.info(
        f"Done training => total of {total_epochs} epochs, {total_batches_processed} mini-batches processed."
    )
    return mean_loss, mean_reward
