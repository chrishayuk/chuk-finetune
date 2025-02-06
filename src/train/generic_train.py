# src/train/generic_train.py
import logging
import numpy as np

# Setup the logger (configure handlers/formatters as needed in your main application)
logger = logging.getLogger(__name__)

def train(
    trainer,
    data_iterator,
    epochs=1,
    batch_size=4,
    log_per_batch=True
):
    """
    A generic training loop that delegates data preparation and training
    steps to a 'trainer' instance. This function doesn’t assume any
    specific training algorithm (SFT, GRPO, etc.) as long as the 'trainer'
    provides the methods:
      1) prepare_batch_data(batch) -> transforms raw data into the trainer's required format
      2) train_step(batch_data) -> returns a scalar loss or a tuple (loss, reward)

    :param trainer: An instance of a Trainer subclass (e.g. GRPOTrainer, SFTTrainer).
    :param data_iterator: A callable or generator that yields batches of data
                          when called with `batch_size` argument, i.e. data_iterator(batch_size).
    :param epochs: Number of epochs to train for.
    :param batch_size: Number of items per mini-batch.
    :param log_per_batch: If True, logs loss/reward at each batch. Otherwise, logs only at epoch boundaries.
    :return: A tuple of (mean_loss, mean_reward).
             For trainers that don't produce a reward, the reward may remain 0.0.
    """
    all_losses = []
    all_rewards = []

    # loop through epochs
    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch + 1}/{epochs}")

        # The data_iterator should yield raw data in batches of size `batch_size`
        for batch_items in data_iterator(batch_size):
            # 1) Transform raw batch into the format that trainer needs
            batch_data = trainer.prepare_batch_data(batch_items)

            # 2) Run a single training step
            result = trainer.train_step(batch_data)
            
            # Handle both single-value (loss) and tuple (loss, reward) returns
            if isinstance(result, tuple):
                loss, reward = result
            else:
                loss, reward = result, None

            # add the losses and rewards
            all_losses.append(loss)
            all_rewards.append(reward if reward is not None else 0.0)

            # Optionally log per-batch results
            if log_per_batch:
                if reward is not None:
                    logger.info(f"Batch => Loss: {loss:.4f}, Reward: {reward:.4f}")
                else:
                    logger.info(f"Batch => Loss: {loss:.4f}")

        # epoch complete
        logger.info(f"Epoch {epoch + 1} complete.")

    # Compute overall statistics
    mean_loss = float(np.mean(all_losses)) if all_losses else 0.0
    mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

    # training complete
    logger.info(
        f"Training complete. "
        f"Mean Loss: {mean_loss:.4f}, Mean Reward: {mean_reward:.4f}"
    )

    # return loss and reward
    return mean_loss, mean_reward