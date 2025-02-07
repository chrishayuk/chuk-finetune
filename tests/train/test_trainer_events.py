import pytest
from pydantic import ValidationError
from train.trainer_events import TrainerEvent

def test_trainer_event_minimal():
    """
    Test creating a TrainerEvent with only event_type.
    This should succeed, and all other fields default to None.
    """
    evt = TrainerEvent(event_type="epoch_start")
    assert evt.event_type == "epoch_start"
    assert evt.epoch is None
    assert evt.batch is None
    assert evt.global_step is None
    assert evt.batch_loss is None
    assert evt.batch_reward is None
    assert evt.epoch_loss is None
    assert evt.epoch_reward is None
    assert evt.mean_loss is None
    assert evt.mean_reward is None

def test_trainer_event_full():
    """
    Test creating a fully populated TrainerEvent.
    """
    evt = TrainerEvent(
        event_type="batch_end",
        epoch=2,
        batch=5,
        global_step=10,
        batch_loss=0.123,
        batch_reward=0.456,
        epoch_loss=0.789,
        epoch_reward=1.234,
        mean_loss=2.345,
        mean_reward=3.456
    )
    assert evt.event_type == "batch_end"
    assert evt.epoch == 2
    assert evt.batch == 5
    assert evt.global_step == 10
    assert evt.batch_loss == 0.123
    assert evt.batch_reward == 0.456
    assert evt.epoch_loss == 0.789
    assert evt.epoch_reward == 1.234
    assert evt.mean_loss == 2.345
    assert evt.mean_reward == 3.456

def test_trainer_event_missing_event_type():
    """
    The 'event_type' field is required. Omission should raise a ValidationError.
    """
    with pytest.raises(ValidationError) as exc_info:
        TrainerEvent()
    assert "event_type" in str(exc_info.value)

def test_trainer_event_invalid_field_type():
    """
    If we pass a string where a float is expected, it should raise ValidationError.
    For example, batch_loss must be a float, so "not_a_float" triggers an error.
    """
    with pytest.raises(ValidationError) as exc_info:
        TrainerEvent(event_type="batch_end", batch_loss="not_a_float")
    assert "batch_loss" in str(exc_info.value)

def test_trainer_event_extra_fields():
    with pytest.raises(ValidationError) as exc_info:
        TrainerEvent(event_type="train_end", unknown_field=123)
    # In Pydantic v2, the message is "Extra inputs are not permitted"
    assert "Extra inputs are not permitted" in str(exc_info.value)
