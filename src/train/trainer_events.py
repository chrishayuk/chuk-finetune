# trainer_events.py
from typing import Optional
from pydantic import BaseModel

class TrainerEvent(BaseModel):
    """
    Represents a single event in the training process.
    
    Attributes:
        event_type (str):
            The type of event, one of:
              - "epoch_start"
              - "batch_end"
              - "epoch_end"
              - "train_end"

        epoch (int, optional):
            Current epoch index (1-based). Valid for epoch-related events.
        batch (int, optional):
            Current batch index (1-based) within the epoch. Valid for batch-related events.
        global_step (int, optional):
            Global training step (increments every time a batch is processed).
        batch_loss (float, optional):
            The loss from the last batch. Valid for "batch_end" events.
        batch_reward (float, optional):
            The reward from the last batch (if applicable). Valid for "batch_end" events.
        epoch_loss (float, optional):
            Mean loss over the entire epoch. Valid for "epoch_end" events.
        epoch_reward (float, optional):
            Mean reward over the entire epoch. Valid for "epoch_end" events.
        mean_loss (float, optional):
            The mean loss over the entire training run (for "train_end" event).
        mean_reward (float, optional):
            The mean reward over the entire training run (for "train_end" event).
    """

    event_type: str
    epoch: Optional[int] = None
    batch: Optional[int] = None
    global_step: Optional[int] = None
    batch_loss: Optional[float] = None
    batch_reward: Optional[float] = None
    epoch_loss: Optional[float] = None
    epoch_reward: Optional[float] = None
    mean_loss: Optional[float] = None
    mean_reward: Optional[float] = None

    class Config:
        # If you have non-primitive fields in the future, you might need:
        # arbitrary_types_allowed = True
        # You can also make the model immutable if desired with 'frozen=True'
        pass
