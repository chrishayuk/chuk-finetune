# tests/train/torch/test_grpo_trainer.py

import pytest
import torch

from train.grpo.torch.grpo_trainer import GRPOTrainer
from tests.fakes.fake_torch_model import FakeTorchModel
from tests.fakes.fake_torch_tokenizer import FakeTorchTokenizer
from tests.fakes.fake_verifier import fake_calculate_reward


def test_grpo_trainer_smoke():
    """
    Smoke test to ensure TorchGRPOTrainer runs end-to-end without error,
    using minimal fake objects.
    """
    device = torch.device("cpu")
    vocab_size = 10
    base_model = FakeTorchModel(vocab_size=vocab_size).to(device)
    ref_model  = FakeTorchModel(vocab_size=vocab_size).to(device)

    tokenizer = FakeTorchTokenizer(vocab_size=vocab_size)

    # Create a small Torch optimizer
    optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)

    # Define a small batch of questions
    batch_questions = [
        "Hello world",
        "What is the meaning of life?",
    ]
    G = 2

    # Create the trainer
    trainer = GRPOTrainer(
        model=base_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        calculate_reward=fake_calculate_reward,
        G=G,
        kl_coeff=0.1,
        device=device,
        verbose=True
    )

    # Prepare batch data
    data_items = [{"prompt": q} for q in batch_questions]
    batch_data = trainer.prepare_batch_data(data_items)

    # Train step
    loss_val, reward_val = trainer.train_step(batch_data)

    # Check results
    assert isinstance(loss_val, float), "Loss should be a float."
    assert torch.isfinite(torch.tensor(loss_val)), "Loss is not finite."
    assert isinstance(reward_val, float), "Reward should be a float."

    print(f"Mean loss from train_step: {loss_val}, mean reward: {reward_val}")
