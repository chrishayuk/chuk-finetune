# src/train/generic_train.py
import logging
import numpy as np

logger = logging.getLogger(__name__)

def generic_train(trainer, data_iterator, epochs=1, batch_size=4):
    """
    A single training loop that ALWAYS yields step-by-step progress:
      - {"epoch_start": True, "epoch": e}
      - {"batch_end": True, "epoch": e, "batch": b, "batch_loss": ..., "batch_reward": ...}
      - {"epoch_end": True, "epoch": e, "epoch_loss": ..., "epoch_reward": ...}
      - {"train_end": True, "mean_loss": ..., "mean_reward": ...}

    It does NOT return anything directly. It's always a generator.

    If you want (mean_loss, mean_reward) at the end, you can iterate over
    this generator, watch for "train_end", and extract the final metrics.
    """
    all_losses = []
    all_rewards = []

    for epoch in range(1, epochs + 1):
        yield {"epoch_start": True, "epoch": epoch}

        epoch_losses = []
        epoch_rewards = []

        for batch_idx, batch_items in enumerate(data_iterator(batch_size), start=1):
            batch_data = trainer.prepare_batch_data(batch_items)
            if not batch_data:
                # skip if empty
                continue

            result = trainer.train_step(batch_data)
            if isinstance(result, tuple):
                loss, reward = result
            else:
                loss, reward = result, None

            epoch_losses.append(loss)
            all_losses.append(loss)
            if reward is not None:
                epoch_rewards.append(reward)
                all_rewards.append(reward)
            else:
                epoch_rewards.append(0.0)
                all_rewards.append(0.0)

            yield {
                "batch_end": True,
                "epoch": epoch,
                "batch": batch_idx,
                "batch_loss": loss,
                "batch_reward": reward if reward is not None else 0.0,
            }

        epoch_mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        epoch_mean_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0

        yield {
            "epoch_end": True,
            "epoch": epoch,
            "epoch_loss": epoch_mean_loss,
            "epoch_reward": epoch_mean_reward,
        }

    # after all epochs
    mean_loss = float(np.mean(all_losses)) if all_losses else 0.0
    mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

    yield {
        "train_end": True,
        "mean_loss": mean_loss,
        "mean_reward": mean_reward
    }