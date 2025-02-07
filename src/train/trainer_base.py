# src/train/trainer.py
from abc import ABC, abstractmethod
from typing import Any, List, Optional

class Trainer(ABC):
    """
    A generic interface for any trainer. Subclasses must implement:
      - `prepare_batch_data(batch)` -> transforms a raw batch into a format usable by `train_step`.
      - `train_step(batch_data)` -> performs a single training step (forward, backward, optimize).

    Subclasses may optionally override these hook methods for logging or extra logic:
      - `on_batch_start(epoch, batch_idx)`  
      - `on_batch_end(epoch, batch_idx, loss, reward)`  
      - `on_epoch_start(epoch)`  
      - `on_epoch_end(epoch, mean_loss, mean_reward)`
      - `on_train_end(mean_loss, mean_reward)`
    """

    def __init__(
        self,
        model: Any,
        tokenizer: Any,
        optimizer: Any,
        device: Optional[str] = None,
        verbose: bool = False
    ):
        """
        Args:
            model (Any): The model to be trained (Torch, MLX, etc.).
            tokenizer (Any): A tokenizer or equivalent component for data processing.
            optimizer (Any): An optimizer object (Torch, MLX, etc.).
            device (str, optional): Device identifier (e.g. "cpu", "cuda", "mlx"). Defaults to None.
            verbose (bool, optional): If True, enables debug/verbose output. Defaults to False.
        """
        self.model = model
        self.tokenizer = tokenizer
        self.optimizer = optimizer
        self.device = device
        self.verbose = verbose

    @abstractmethod
    def prepare_batch_data(self, batch: List[Any]) -> Any:
        """
        Takes a raw batch of data (list of examples) and returns a processed
        structure suitable for train_step.

        Args:
            batch (List[Any]): A list of raw data items (e.g., dictionaries, strings, etc.).

        Returns:
            Any: A processed form of the batch suitable for `train_step` (could be a dict,
                 a custom dataclass, etc.). The exact structure is up to the subclass.
        """
        raise NotImplementedError()

    @abstractmethod
    def train_step(self, batch_data: Any) -> Any:
        """
        Performs one training step (forward pass, compute loss, backprop, optimizer step).
        Must return relevant metrics, typically (loss, reward).

        Args:
            batch_data (Any): The processed batch data returned by `prepare_batch_data`.

        Returns:
            Any: Typically a tuple (loss, reward) or similar. If reward is not applicable,
                 you can return (loss, 0.0) or just `loss`. The calling loop will adapt.
        """
        raise NotImplementedError()

    def on_batch_start(self, epoch: int, batch_idx: int) -> None:
        """
        Called before each batch is processed. Subclasses can override to perform
        batch-level logging or pre-processing.

        Args:
            epoch (int): Current epoch (1-based).
            batch_idx (int): Current batch index within the epoch (1-based).
        """
        pass

    def on_batch_end(self, epoch: int, batch_idx: int, loss: float, reward: float) -> None:
        """
        Called after each batch is processed. Subclasses can override to perform
        logging or other side effects.

        Args:
            epoch (int): Current epoch (1-based).
            batch_idx (int): Current batch index within the epoch (1-based).
            loss (float): The loss returned by train_step.
            reward (float): The reward returned by train_step (or 0 if not applicable).
        """
        pass

    def on_epoch_start(self, epoch: int) -> None:
        """
        Called at the start of each epoch.

        Args:
            epoch (int): Current epoch (1-based).
        """
        pass

    def on_epoch_end(self, epoch: int, mean_loss: float, mean_reward: float) -> None:
        """
        Called at the end of each epoch.

        Args:
            epoch (int): Current epoch (1-based).
            mean_loss (float): Mean loss for the entire epoch.
            mean_reward (float): Mean reward for the entire epoch.
        """
        pass

    def on_train_end(self, mean_loss: float, mean_reward: float) -> None:
        """
        Called after all epochs (or after hitting max_steps) when training completes.

        Args:
            mean_loss (float): Mean loss over the entire training run.
            mean_reward (float): Mean reward over the entire training run.
        """
        pass
