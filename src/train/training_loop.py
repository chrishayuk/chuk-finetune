import logging
import numpy as np

# setup the logger
logger = logging.getLogger(__name__)

def train(
    trainer,
    data_iterator,
    epochs=1,
    batch_size=4
):
    """
    Generic training loop that delegates training to the methods of 'trainer'.
    The 'trainer' object can be an SFTTrainer, a GRPOTrainer, or any other
    that follows the same interface.
    """
    all_losses = []
    all_rewards = []

    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch+1}/{epochs}")

        # data_iterator should yield batches of size `batch_size`
        for batch in data_iterator(batch_size):
            # 1) Prepare data for the trainer
            batch_data = trainer.prepare_batch_data(batch)

            # 2) Train step (get loss + optional reward)
            result = trainer.train_step(batch_data)

            # We expect result to be something like (loss, reward)
            if isinstance(result, tuple):
                loss, reward = result
            else:
                # If trainer returns just a scalar
                loss, reward = result, None

            all_losses.append(loss)
            if reward is not None:
                all_rewards.append(reward)

            logger.info(f"Batch => Loss: {loss:.4f}"
                        f"{f', Reward: {reward:.4f}' if reward is not None else ''}")

    mean_loss = float(np.mean(all_losses)) if all_losses else 0.0
    mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

    logger.info(f"Training complete. Mean Loss: {mean_loss:.4f}, Mean Reward: {mean_reward:.4f}")
    return mean_loss, mean_reward
