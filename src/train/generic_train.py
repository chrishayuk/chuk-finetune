# src/train/generic_train.py
import logging
import numpy as np

# setup the logger
logger = logging.getLogger(__name__)

def train(trainer, data_iterator, epochs=1):
    """
    Generic training loop that calls:
      1) trainer.prepare_batch_data(batch)
      2) trainer.train_step(batch_data)

    across a specified number of epochs. 
    The data_iterator should yield batches of size 'batch_size'.
    This function doesn't care about the specific training algorithm 
    (SFT, GRPO, etc.) as long as the trainer provides the required interface.

    :param trainer: An instance of a Trainer subclass (e.g. GRPOTrainer, SFTTrainer).
    :param data_iterator: A callable or generator that yields batches of data 
                          when called with a 'batch_size' argument.
    :param epochs: Number of epochs to train for.
    :param batch_size: Number of items per mini-batch.
    :return: A tuple of (mean_loss, mean_reward). 
             For trainers that don't produce a reward, the reward might remain 0.0.
    """
    all_losses = []
    all_rewards = []

    for epoch in range(epochs):
        logger.info(f"Starting epoch {epoch+1}/{epochs}")

        # data_iterator(batch_size) should yield batches of raw items
        for batch_questions in data_iterator():
            # 1) Transform raw batch into the structure trainer needs
            batch_data = trainer.prepare_batch_data(batch_questions)

            # 2) Perform a single training step
            loss, reward = trainer.train_step(batch_data)

            # get the losses for the batch
            all_losses.append(loss)
            all_rewards.append(reward if reward is not None else 0.0)

    # Compute overall statistics
    mean_loss = float(np.mean(all_losses)) if all_losses else 0.0
    mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

    # log it
    logger.info(f"Training complete. Mean Loss: {mean_loss:.4f}, Mean Reward: {mean_reward:.4f}")

    # return the loss and reward
    return mean_loss, mean_reward
