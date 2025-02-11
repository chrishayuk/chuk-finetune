# tests/train/test_grpo_trainer.py

import pytest
from unittest.mock import patch, MagicMock

from train.grpo.grpo_trainer import train_grpo


@pytest.mark.parametrize("device_override,expected_framework,is_mlx",
    [
        ("mlx", "mlx", True),
        ("cpu", "torch", False),
        # Could add more like ("cuda", "torch", False), etc.
    ]
)
@pytest.mark.parametrize("as_generator", [False, True])
def test_train_grpo(
    device_override,
    expected_framework,
    is_mlx,
    as_generator
):
    """
    Tests the train_grpo function with different devices and as_generator modes.
    We mock out get_optimizer, get_dataloader, and generic_train so no real training occurs.
    """

    # 1) Mocks for get_optimizer, get_dataloader, generic_train
    mock_optimizer = MagicMock(name="optimizer")
    mock_data_iterator_fn = MagicMock(name="data_iterator_fn")

    # We'll make generic_train return a generator that yields one event with event_type="train_end"
    # and some final mean_loss, mean_reward
    mock_event = MagicMock(name="TrainerEvent")
    mock_event.event_type = "train_end"
    mock_event.mean_loss = 42.0
    mock_event.mean_reward = 3.14

    def fake_generator():
        # yield one event
        yield mock_event

    # We'll patch them in the correct module path
    with patch("train.grpo.grpo_trainer.get_optimizer", return_value=mock_optimizer) as mock_get_optimizer, \
         patch("train.grpo.grpo_trainer.get_dataloader", return_value=mock_data_iterator_fn) as mock_get_dataloader, \
         patch("train.grpo.grpo_trainer.generic_train", return_value=fake_generator()) as mock_generic_train:

        # 2) Prepare minimal arguments
        base_model = MagicMock(name="base_model")
        ref_model = MagicMock(name="ref_model")
        tokenizer = MagicMock(name="tokenizer")
        dataset = MagicMock(name="dataset")
        calculate_reward = MagicMock(name="calculate_reward")
        lr = 1e-4

        # 3) Call the function under test
        result = train_grpo(
            base_model=base_model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            dataset=dataset,
            calculate_reward=calculate_reward,
            lr=lr,
            epochs=2,
            batch_size=4,
            G=4,
            device=device_override,
            verbose=True,
            kl_coeff=0.1,
            as_generator=as_generator
        )

        # 4) Verify the calls to the mocks

        # get_optimizer => (framework, base_model, lr=...)
        mock_get_optimizer.assert_called_once()
        args, kwargs = mock_get_optimizer.call_args
        assert args[0] == expected_framework
        assert args[1] is base_model
        assert kwargs["lr"] == lr

        # get_dataloader => called with (framework, dataset, batch_size, shuffle=True)
        mock_get_dataloader.assert_called_once()
        args, kwargs = mock_get_dataloader.call_args
        assert args[0] == expected_framework
        assert args[1] is dataset

        # Since batch_size is passed as a positional arg (third),
        # we check args[2], not kwargs["batch_size"].
        assert args[2] == 4
        # 'shuffle=True' is a named kwarg
        assert kwargs["shuffle"] is True

        # generic_train => called once => generator of events
        mock_generic_train.assert_called_once()

        # 5) Check result
        if as_generator:
            # If as_generator=True => returns the generator
            gen = result
            assert gen is not None
            events = list(gen)  # consume
            assert len(events) == 1
            assert events[0].mean_loss == 42.0
            assert events[0].mean_reward == 3.14
        else:
            # If as_generator=False => returns (mean_loss, mean_reward)
            mean_loss, mean_reward = result
            assert mean_loss == 42.0
            assert mean_reward == 3.14

        # 6) Additional checks regarding MLX vs. Torch
        if is_mlx:
            # We imported GRPOTrainer from train.grpo.mlx.grpo_trainer
            pass
        else:
            # "torch"
            pass
