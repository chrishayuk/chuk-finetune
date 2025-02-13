import pytest
import logging
from unittest.mock import MagicMock
import torch

from train.teacher.run_teacher_collection import collect_teacher_data_once

logger = logging.getLogger(__name__)

def generate_teacher_mock(prompt, verbose=False, max_new_tokens=1024, temperature=0.7, top_p=0.95):
    """
    Mock teacher generation: returns (prompt + ' => teacher_out', logprob=1.23).
    The signature now matches the unified teacher generation function.
    """
    response_text = prompt + " => teacher_out"
    return (response_text, 1.23)

def calculate_reward_mock(resp_text, item_dict):
    """
    If 'skip' is in the prompt, return (None, ""), causing the item to be skipped.
    Otherwise, return (1.0, ""), accepting the item with a reward of 1.0.
    """
    if "skip" in item_dict["prompt"].lower():
        return (None, "")
    return (1.0, "")

@pytest.fixture
def mock_teacher_model():
    """
    A minimal placeholder representing a teacher model.
    For testing, we return a MagicMock.
    """
    return MagicMock(name="mock_teacher_model")

class FakeTokenizer:
    eos_token_id = 999

    def __call__(self, text, return_tensors="pt"):
        # Return a real torch tensor with shape [1,3]
        return {"input_ids": torch.tensor([[101, 102, 103]])}
    
    def encode(self, text):
        return [101, 102, 103]

@pytest.fixture
def mock_tokenizer():
    """
    A minimal tokenizer placeholder that returns real tensors for Torch.
    """
    return FakeTokenizer()

def test_collect_teacher_data_once_basic(mock_teacher_model, mock_tokenizer, caplog):
    """
    Tests 'collect_teacher_data_once' with a small dataset,
    ensuring that items with 'skip' in the prompt are omitted.
    We pass device='torch' so that the Torch branch is used.
    """
    dataset = [
        {"prompt": "Question1: do something."},
        {"prompt": "Question2: skip this one please."},
        {"prompt": "Question3: final question."},
    ]

    final_data = collect_teacher_data_once(
        teacher_model=mock_teacher_model,
        tokenizer=mock_tokenizer,
        dataset=dataset,
        calculate_reward=calculate_reward_mock,
        batch_size=2,
        G=1,
        device="torch",  # Use Torch logic
        verbose=True,
        generate_single_fn=generate_teacher_mock
    )

    # Expect that the second item is skipped (because of "skip" in prompt)
    assert len(final_data) == 2

    item1 = final_data[0]
    # For G=1, expect one response, one teacher log-prob, one reward.
    assert "responses" in item1 and len(item1["responses"]) == 1
    assert "teacher_logprobs" in item1 and len(item1["teacher_logprobs"]) == 1
    assert "rewards" in item1 and len(item1["rewards"]) == 1
    # Check the returned logprob and reward values.
    assert item1["teacher_logprobs"][0] == 1.23
    assert item1["rewards"][0] == 1.0

    # Ensure that none of the final items contain "skip" in their prompt.
    prompts_in_result = [x["item"]["prompt"] for x in final_data]
    assert all("skip" not in prompt.lower() for prompt in prompts_in_result)

def test_collect_teacher_data_once_empty(mock_teacher_model, mock_tokenizer):
    """
    Tests the single-pass approach with an empty dataset; no items should be collected.
    """
    dataset = []

    final_data = collect_teacher_data_once(
        teacher_model=mock_teacher_model,
        tokenizer=mock_tokenizer,
        dataset=dataset,
        calculate_reward=calculate_reward_mock,
        batch_size=2,
        G=1,
        device="torch",
        verbose=True,
        generate_single_fn=generate_teacher_mock
    )
    assert len(final_data) == 0
