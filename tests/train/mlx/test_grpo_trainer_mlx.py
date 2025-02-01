# tests/test_grpo_trainer_mlx.py
import pytest
import mlx.core as mx

from train.mlx.grpo_trainer import train_step
from tests.fakes.fake_mlx_model import FakeMLXModel
from tests.fakes.fake_mlx_tokenizer import FakeMLXTokenizer
from tests.fakes.fake_verifier import FakeVerifier, fake_calculate_reward

def test_train_step_smoke():
    """
    Smoke test for MLX train_step(...) verifying it runs end-to-end with minimal objects.
    """
    # 1) Create a fake MLX model (base) + a fake MLX model (ref)
    vocab_size = 10
    base_model = FakeMLXModel(vocab_size)
    ref_model  = FakeMLXModel(vocab_size)

    # 2) Fake MLX tokenizer
    tokenizer = FakeMLXTokenizer(vocab_size=vocab_size)

    # 3) Fake verifier
    verifier = FakeVerifier()

    # 4) Create an MLX optimizer
    import mlx.optimizers as optim
    optimizer = optim.Adam(learning_rate=1e-3)

    # 5) A small batch of questions
    batch_questions = [
        "Hello MLX",
        "This is good or not?",
    ]
    G = 2

    # 6) Call train_step
    loss_val = train_step(
        base_model=base_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        batch_questions=batch_questions,
        verifier=verifier,
        G=G,
        optimizer=optimizer,
        calculate_reward=fake_calculate_reward,
        device=None,   # MLX typically ignores device
        verbose=True
    )

    # 7) Confirm it returns a float
    assert isinstance(loss_val, float), "train_step should return a float loss"
    print(f"MLX train_step smoke test => loss: {loss_val}")
