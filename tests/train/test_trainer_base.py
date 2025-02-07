# tests/test_trainer_base.py
import pytest
from abc import ABC, abstractmethod
from train.trainer_base import Trainer

def test_trainer_cannot_instantiate():
    """
    Because Trainer has abstract methods (prepare_batch_data, train_step),
    attempting to instantiate it directly should raise TypeError.
    """
    with pytest.raises(TypeError) as exc_info:
        Trainer(model=None, tokenizer=None, optimizer=None)
    assert "Can't instantiate abstract class Trainer" in str(exc_info.value)


class DummyTrainer(Trainer):
    """
    A minimal concrete subclass of Trainer that implements the abstract methods.
    """

    def prepare_batch_data(self, batch):
        # minimal implementation
        return batch

    def train_step(self, batch_data):
        # minimal implementation
        return 0.0, 0.0  # loss, reward


def test_trainer_subclass_instantiation():
    """
    Ensure we can instantiate a subclass that implements the abstract methods.
    """
    trainer = DummyTrainer(
        model="dummy_model",
        tokenizer="dummy_tokenizer",
        optimizer="dummy_optimizer",
        device="cpu",
        verbose=True
    )
    assert trainer.model == "dummy_model"
    assert trainer.tokenizer == "dummy_tokenizer"
    assert trainer.optimizer == "dummy_optimizer"
    assert trainer.device == "cpu"
    assert trainer.verbose is True


def test_trainer_subclass_methods():
    """
    Check that the subclass methods can be called without error.
    """
    trainer = DummyTrainer(
        model=None,
        tokenizer=None,
        optimizer=None,
        device=None,
        verbose=False
    )

    # We'll define a sample batch
    sample_batch = ["data1", "data2"]
    batch_data = trainer.prepare_batch_data(sample_batch)
    assert batch_data == sample_batch, "Expected prepare_batch_data to return the input in this dummy implementation"

    loss, reward = trainer.train_step(batch_data)
    assert loss == 0.0
    assert reward == 0.0


def test_trainer_subclass_hooks():
    """
    Ensure hook methods can be called. They do nothing by default, but let's verify no errors.
    """
    trainer = DummyTrainer(
        model=None,
        tokenizer=None,
        optimizer=None,
        device=None,
        verbose=False
    )

    # We don't expect these calls to raise an error by default
    trainer.on_epoch_start(epoch=1)
    trainer.on_batch_start(epoch=1, batch_idx=42)
    trainer.on_batch_end(epoch=1, batch_idx=42, loss=1.234, reward=0.567)
    trainer.on_epoch_end(epoch=1, mean_loss=0.999, mean_reward=0.111)
    trainer.on_train_end(mean_loss=0.888, mean_reward=0.222)

    # If you want to confirm no side effects, there's nothing to assert here
    # because the default hooks do nothing. But in a real subclass you might store logs, etc.
