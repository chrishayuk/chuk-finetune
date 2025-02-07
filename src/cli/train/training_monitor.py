# src/cli/train/training_monitor.py
import logging

# imports
from cli.train.logger_config import YELLOW, GREEN, color_text
from train.trainer_events import TrainerEvent  # or wherever your event model is defined

logger = logging.getLogger(__name__)

def monitor_training_progress(gen) -> (float, float):
    """
    Consumes events from a generator-based training loop.
    Each event is a TrainerEvent (a Pydantic model or dataclass) with:
        event_type: str  -> "epoch_start", "batch_end", "epoch_end", "train_end"
    plus optional fields like:
        epoch, batch, batch_loss, batch_reward, epoch_loss, epoch_reward, mean_loss, mean_reward

    Returns:
        (final_mean_loss, final_mean_reward) at the end of training.
    """
    final_mean_loss = 0.0
    final_mean_reward = 0.0

    for event in gen:
        # event is a TrainerEvent
        if event.event_type == "epoch_start":
            logger.info(f"\n--- Starting epoch {event.epoch} ---")

        elif event.event_type == "batch_end":
            # event.batch, event.batch_loss, event.batch_reward
            if event.batch_loss is not None:
                logger.info(color_text(
                    f"Batch {event.batch} ended => loss={event.batch_loss:.4f}",
                    YELLOW
                ))

        elif event.event_type == "epoch_end":
            # event.epoch, event.epoch_loss, event.epoch_reward
            epoch_loss = event.epoch_loss or 0.0
            logger.info(color_text(
                f"=== Finished epoch {event.epoch} => mean_loss={epoch_loss:.4f}",
                GREEN
            ))

        elif event.event_type == "train_end":
            # event.mean_loss, event.mean_reward
            final_mean_loss = event.mean_loss or 0.0
            final_mean_reward = event.mean_reward or 0.0
            logger.info(color_text(
                f"Training complete => mean_loss={final_mean_loss:.4f}, "
                f"mean_reward={final_mean_reward:.4f}",
                GREEN
            ))

        else:
            # Handle any custom event types if your trainer yields them
            pass

    # Return final metrics 
    return final_mean_loss, final_mean_reward
