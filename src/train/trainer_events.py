# trainer_events.py
from typing import Optional
from pydantic import BaseModel, ConfigDict

class TrainerEvent(BaseModel):
    """
    Represents a single event in the training process.
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

    # Pydantic V2 style config
    model_config = ConfigDict(extra='forbid')
