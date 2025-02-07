# src/cli/train/training_monitor.py
import logging

# imports
from cli.train.logger_config import YELLOW, GREEN, color_text

# logger
logger = logging.getLogger(__name__)

def monitor_training_progress(gen):
    """
    Consumes events from a generator-based training loop.
    Each event is a dict that may contain:
      - "epoch_start"
      - "batch_end"
      - "epoch_end"
      - "train_end"
    and their associated data (e.g., loss, reward).

    Returns the final mean_loss (if any).
    """
    final_mean_loss = 0.0

    for event in gen:
        if "epoch_start" in event:
            logger.info(f"\n--- Starting epoch {event['epoch']} ---")

        elif "batch_end" in event:
            logger.info(color_text(
                f"Batch {event['batch']} ended with mean loss={event['batch_loss']:.4f}",
                YELLOW
            ))

        elif "epoch_end" in event:
            epoch_loss = event.get("epoch_loss", 0.0)
            logger.info(color_text(
                f"=== Finished epoch {event['epoch']} -> mean_loss={epoch_loss:.4f}",
                GREEN
            ))

        elif "train_end" in event:
            final_mean_loss = event.get("mean_loss", 0.0)
            mean_reward = event.get("mean_reward", 0.0)
            logger.info(color_text(
                f"Training complete => mean_loss={final_mean_loss:.4f}, mean_reward={mean_reward:.4f}",
                GREEN
            ))

        else:
            # If your training loop yields other custom keys or data,
            # you can handle them here.
            pass

    # return final mean, loss
    return final_mean_loss
