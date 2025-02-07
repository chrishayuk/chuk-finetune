# src/train/generic_train.py
import logging
import numpy as np
from typing import Generator, Callable, Optional

from train.trainer_base import Trainer
from train.trainer_events import TrainerEvent

logger = logging.getLogger(__name__)

def generic_train(
    trainer: Trainer,
    data_iterator: Callable[[int], Generator],
    epochs: int = 1,
    batch_size: int = 4,
    max_steps: Optional[int] = None
) -> Generator[TrainerEvent, None, None]:
    """
    A single training loop that can terminate based on epochs or a fixed number of steps.
    Yields pydantic-based TrainerEvent objects instead of raw dictionaries.
    """

    # all counters counters
    all_losses = []
    all_rewards = []
    global_step = 0
    done = False

    # loop through each epoch
    for epoch in range(1, epochs + 1):
        # on epoch start hook
        trainer.on_epoch_start(epoch)

        # yield a trainer event
        yield TrainerEvent(event_type="epoch_start", epoch=epoch)

        # reset epoch level counters
        epoch_losses = []
        epoch_rewards = []

        # data_iterator yields batches
        for batch_idx, batch_items in enumerate(data_iterator(batch_size), start=1):
            # HOOK
            trainer.on_batch_start(epoch, batch_idx)

            # prepare the batch data
            batch_data = trainer.prepare_batch_data(batch_items)

            # no batch data
            if not batch_data:
                # skip if empty
                continue

            # perform a training step
            result = trainer.train_step(batch_data)

            # check we got a result
            if isinstance(result, tuple):
                loss, reward = result
            else:
                loss, reward = result, None

            # add losses to epoch and global counts
            epoch_losses.append(loss)
            all_losses.append(loss)

            # check for a reward
            if reward is None:
                reward = 0.0

            # add rewords to epoch and global counts
            epoch_rewards.append(reward)
            all_rewards.append(reward)

            # increase our step count
            global_step += 1

            # batch ended
            trainer.on_batch_end(epoch, batch_idx, loss, reward)

            # yield the trainer batch event
            yield TrainerEvent(
                event_type="batch_end",
                epoch=epoch,
                batch=batch_idx,
                global_step=global_step,
                batch_loss=loss,
                batch_reward=reward
            )

            # Check if we've hit the max_steps limit
            if max_steps is not None and global_step >= max_steps:
                done = True
                break

        # end of this epoch
        epoch_mean_loss = float(np.mean(epoch_losses)) if epoch_losses else 0.0
        epoch_mean_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0

        # epoch ended
        trainer.on_epoch_end(epoch, epoch_mean_loss, epoch_mean_reward)

        # yield the trainer event
        yield TrainerEvent(
            event_type="epoch_end",
            epoch=epoch,
            epoch_loss=epoch_mean_loss,
            epoch_reward=epoch_mean_reward
        )

        # are we done
        if done:
            break

    # after all epochs (or max_steps)
    mean_loss = float(np.mean(all_losses)) if all_losses else 0.0
    mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

    # training ended
    trainer.on_train_end(mean_loss, mean_reward)

    # yield the trainer training event
    yield TrainerEvent(
        event_type="train_end",
        mean_loss=mean_loss,
        mean_reward=mean_reward
    )
