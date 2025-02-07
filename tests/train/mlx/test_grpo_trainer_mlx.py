# tests/test_grpo_trainer_mlx.py
import pytest
import mlx.core as mx
import mlx.optimizers as optim

from train.grpo.mlx.grpo_trainer import GRPOTrainer
from tests.fakes.fake_mlx_model import FakeMLXModel
from tests.fakes.fake_mlx_tokenizer import FakeMLXTokenizer
from tests.fakes.fake_verifier import fake_calculate_reward


def test_train_step_smoke():
    """
    Smoke test for MLX GRPOTrainer - verifying end-to-end functionality with minimal objects.
    """
    # 1) Create a fake MLX model (base) + a fake MLX model (ref)
    vocab_size = 10
    base_model = FakeMLXModel(vocab_size)
    ref_model  = FakeMLXModel(vocab_size)

    # 2) Fake MLX tokenizer
    tokenizer = FakeMLXTokenizer(vocab_size=vocab_size)

    # 3) Create an MLX optimizer
    optimizer = optim.Adam(learning_rate=1e-3)

    # 4) A small batch of "items" containing both "prompt" and "verifiers"
    data_items = [
        {
            "prompt": "Hello MLX",
            "verifiers": [{}],  # minimal fake
        },
        {
            "prompt": "Is this good or not?",
            "verifiers": [],    # no verifiers
        }
    ]
    G = 2

    # 5) Create the trainer instance
    trainer = GRPOTrainer(
        model=base_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        optimizer=optimizer,
        calculate_reward=fake_calculate_reward,  # now handles (resp, item)
        G=G,
        kl_coeff=0.1,
        device=None,
        verbose=True
    )

    # 6) Prepare the batch data
    batch_data = trainer.prepare_batch_data(data_items)

    # 7) Run the train_step
    loss_val, reward_val = trainer.train_step(batch_data)

    # 8) Confirm it returns floats
    assert isinstance(loss_val, float), "train_step should return a float loss"
    assert isinstance(reward_val, float), "train_step should return a float reward"

    print(f"GRPOTrainer smoke test => loss: {loss_val}, reward: {reward_val}")
