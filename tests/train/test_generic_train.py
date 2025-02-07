# tests/test_generic_train.py

import pytest
from typing import Generator, List

from train.generic_train import generic_train
from train.trainer_base import Trainer
from train.trainer_events import TrainerEvent

class DummyTrainer(Trainer):
    """
    A minimal trainer used for testing generic_train.
    We override prepare_batch_data and train_step with trivial logic.
    """
    def __init__(self):
        # Just pass None for model, tokenizer, optimizer
        super().__init__(model=None, tokenizer=None, optimizer=None)
        # We'll track how many times hooks are called
        self.epoch_starts = []
        self.epoch_ends = []
        self.batch_starts = []
        self.batch_ends = []
        self.train_end_called = False

    def prepare_batch_data(self, batch):
        # For testing, just echo back the batch if non-empty
        if batch:
            return batch  # return as "batch_data"
        return []

    def train_step(self, batch_data):
        # Let's pretend each batch is a list of "N" items => loss = N, reward = N * 0.5
        n = len(batch_data)
        return float(n), float(n) * 0.5

    # Hooks
    def on_epoch_start(self, epoch):
        self.epoch_starts.append(epoch)

    def on_epoch_end(self, epoch, mean_loss, mean_reward):
        self.epoch_ends.append((epoch, mean_loss, mean_reward))

    def on_batch_start(self, epoch, batch_idx):
        self.batch_starts.append((epoch, batch_idx))

    def on_batch_end(self, epoch, batch_idx, loss, reward):
        self.batch_ends.append((epoch, batch_idx, loss, reward))

    def on_train_end(self, mean_loss, mean_reward):
        self.train_end_called = True


def make_data_iterator(num_batches: int, batch_size: int) -> Generator[List[int], None, None]:
    """
    Creates a simple data iterator that yields `num_batches` times,
    each batch is a list of size `batch_size`.
    We'll just fill it with dummy integers (0..).
    """
    start = 0
    for _ in range(num_batches):
        yield list(range(start, start + batch_size))
        start += batch_size


def test_generic_train_no_data():
    """
    Test the case where the data iterator yields no batches.
    """
    trainer = DummyTrainer()
    
    def empty_iterator(batch_size: int):
        # yields nothing
        if False:
            yield

    gen = generic_train(
        trainer=trainer,
        data_iterator=empty_iterator,
        epochs=2,
        batch_size=4,
        max_steps=None
    )

    events = list(gen)  # consume all events

    # We should see 2 epochs start, then 2 epoch_end, then train_end
    # But no batch_end events.
    epoch_start_events = [e for e in events if e.event_type == "epoch_start"]
    epoch_end_events = [e for e in events if e.event_type == "epoch_end"]
    train_end_events = [e for e in events if e.event_type == "train_end"]
    batch_end_events = [e for e in events if e.event_type == "batch_end"]

    assert len(epoch_start_events) == 2
    assert len(epoch_end_events) == 2
    assert len(train_end_events) == 1
    assert len(batch_end_events) == 0

    # Because no data => mean_loss and mean_reward should be 0.0
    assert train_end_events[0].mean_loss == 0.0
    assert train_end_events[0].mean_reward == 0.0

    # Check trainer hooks
    assert trainer.epoch_starts == [1, 2]
    # Each epoch had no batches => on_epoch_end triggers but with 0.0 stats
    assert trainer.epoch_ends[0][1] == 0.0  # mean_loss in epoch1
    assert trainer.epoch_ends[0][2] == 0.0  # mean_reward in epoch1
    assert trainer.epoch_ends[1][1] == 0.0  # epoch2
    assert trainer.epoch_ends[1][2] == 0.0
    assert trainer.train_end_called is True


def test_generic_train_one_epoch():
    """
    Test a single epoch with some data, no max_steps.
    """
    trainer = DummyTrainer()

    # We'll yield 2 batches
    def data_iterator(batch_size: int):
        yield [1,2]   # batch of size 2
        yield [3,4,5] # batch of size 3

    gen = generic_train(
        trainer=trainer,
        data_iterator=data_iterator,
        epochs=1,   # one epoch
        batch_size=4
    )

    events = list(gen)

    epoch_start_events = [e for e in events if e.event_type == "epoch_start"]
    batch_end_events = [e for e in events if e.event_type == "batch_end"]
    epoch_end_events = [e for e in events if e.event_type == "epoch_end"]
    train_end_events = [e for e in events if e.event_type == "train_end"]

    assert len(epoch_start_events) == 1
    assert len(batch_end_events) == 2
    assert len(epoch_end_events) == 1
    assert len(train_end_events) == 1

    # Check correctness of batch_end events
    # First batch had size=2 => loss=2, reward=1.0
    # Second batch had size=3 => loss=3, reward=1.5
    assert batch_end_events[0].batch_loss == 2.0
    assert batch_end_events[0].batch_reward == 1.0
    assert batch_end_events[1].batch_loss == 3.0
    assert batch_end_events[1].batch_reward == 1.5

    # Epoch mean loss should be (2 + 3)/2 = 2.5
    # reward = (1.0 + 1.5)/2 = 1.25
    assert epoch_end_events[0].epoch_loss == pytest.approx(2.5)
    assert epoch_end_events[0].epoch_reward == pytest.approx(1.25)

    # Final train_end
    # overall mean_loss => also 2.5
    # overall mean_reward => 1.25
    assert train_end_events[0].mean_loss == pytest.approx(2.5)
    assert train_end_events[0].mean_reward == pytest.approx(1.25)

    # Check trainer hooks
    # We had 1 epoch => epoch_start(1), epoch_end(1)
    assert trainer.epoch_starts == [1]
    assert trainer.epoch_ends[0][0] == 1  # epoch number
    assert trainer.epoch_ends[0][1] == pytest.approx(2.5)  # mean_loss
    assert trainer.epoch_ends[0][2] == pytest.approx(1.25) # mean_reward

    # Batches:
    #  - on_batch_start(1,1), on_batch_end(1,1, 2.0, 1.0)
    #  - on_batch_start(1,2), on_batch_end(1,2, 3.0, 1.5)
    assert len(trainer.batch_starts) == 2
    assert trainer.batch_starts[0] == (1, 1)
    assert trainer.batch_starts[1] == (1, 2)

    assert len(trainer.batch_ends) == 2
    assert trainer.batch_ends[0] == (1, 1, 2.0, 1.0)
    assert trainer.batch_ends[1] == (1, 2, 3.0, 1.5)

    assert trainer.train_end_called is True


def test_generic_train_max_steps():
    """
    Test stopping early with max_steps = 2 across multiple epochs.
    We'll define enough data for 4 batches, but we only want 2 steps total.
    """
    trainer = DummyTrainer()

    # We'll yield 2 batches per epoch => total 4 across 2 epochs
    # But we'll set max_steps=2 => only the first 2 batches overall
    def data_iterator(batch_size: int):
        # first call => yield [1,2]
        yield [1,2]
        # second call => yield [3]
        yield [3]
        # third => yield [4,5]
        yield [4,5]
        # fourth => yield [6,7]
        yield [6,7]

    gen = generic_train(
        trainer=trainer,
        data_iterator=data_iterator,
        epochs=2,
        batch_size=3,  # doesn't matter too much, we'll just get the above lumps
        max_steps=2
    )

    events = list(gen)

    epoch_start_events = [e for e in events if e.event_type == "epoch_start"]
    batch_end_events = [e for e in events if e.event_type == "batch_end"]
    epoch_end_events = [e for e in events if e.event_type == "epoch_end"]
    train_end_events = [e for e in events if e.event_type == "train_end"]

    # We expect:
    #   - epoch_start(1)
    #   - batch_end(1,1)
    #   - batch_end(1,2)  => after this, max_steps=2 => done
    #   - epoch_end(1)    => because we break out of batch loop
    #   - train_end
    # => We do NOT start epoch 2
    assert len(epoch_start_events) == 1
    assert epoch_start_events[0].epoch == 1

    assert len(batch_end_events) == 2
    # first batch => size=2 => loss=2, reward=1.0
    # second batch => size=1 => loss=1, reward=0.5
    assert batch_end_events[0].batch_loss == 2.0
    assert batch_end_events[0].batch_reward == 1.0
    assert batch_end_events[1].batch_loss == 1.0
    assert batch_end_events[1].batch_reward == 0.5

    assert len(epoch_end_events) == 1
    assert epoch_end_events[0].epoch_loss == pytest.approx(1.5)  # avg(2.0, 1.0) = 1.5
    assert epoch_end_events[0].epoch_reward == pytest.approx(0.75)

    assert len(train_end_events) == 1
    assert train_end_events[0].mean_loss == pytest.approx(1.5)
    assert train_end_events[0].mean_reward == pytest.approx(0.75)

    # Check hooks
    # We only started epoch1 => on_epoch_start(1)
    # 2 batches => (1,1), (1,2)
    # then break => on_epoch_end(1)
    assert trainer.epoch_starts == [1]
    assert trainer.epoch_ends == [(1, 1.5, 0.75)]
    assert trainer.batch_starts == [(1,1), (1,2)]
    assert trainer.batch_ends == [
        (1,1, 2.0,1.0),
        (1,2, 1.0,0.5)
    ]
    assert trainer.train_end_called is True
