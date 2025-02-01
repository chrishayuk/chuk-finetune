# tests/train/torch/test_grpo_trainer.py

import pytest
import torch

from src.train.torch.grpo_trainer import train_step

# Our fakes
from tests.fakes.fake_torch_model import FakeTorchModel
from tests.fakes.fake_torch_tokenizer import FakeTorchTokenizer
from tests.fakes.fake_verifier import FakeVerifier, fake_calculate_reward

def test_train_step_smoke():
    """
    Smoke test to ensure train_step(...) runs end-to-end without error,
    using minimal fake objects.
    """
    device = torch.device("cpu")
    vocab_size = 10
    base_model = FakeTorchModel(vocab_size=vocab_size).to(device)
    ref_model  = FakeTorchModel(vocab_size=vocab_size).to(device)

    tokenizer = FakeTorchTokenizer(vocab_size=vocab_size)
    verifier = FakeVerifier()

    # create a small Torch optimizer
    optimizer = torch.optim.Adam(base_model.parameters(), lr=1e-3)

    # define a small batch of questions
    batch_questions = [
        "Hello world",
        "What is the meaning of life?",
    ]
    G = 2

    loss_val = train_step(
        base_model=base_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        batch_questions=batch_questions,
        verifier=verifier,
        G=G,
        optimizer=optimizer,
        calculate_reward=fake_calculate_reward,
        device=device,
        verbose=True
    )
    # Just check it's a finite float
    assert isinstance(loss_val, float)
    assert torch.isfinite(torch.tensor(loss_val)), "Loss is not finite"

    print(f"Mean loss from train_step: {loss_val}")
