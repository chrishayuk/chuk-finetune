import pytest
import logging

from train.teacher.run_teacher_collection import collect_teacher_data_once
from train.teacher.teacher_trainer import TeacherTrainer

def generate_teacher_mock(prompt, verbose=False):
    """
    Mock teacher generation: returns (prompt + ' => teacher_out', logprob=1.23).
    """
    response_text = prompt + " => teacher_out"
    return (response_text, 1.23)

def calculate_reward_mock(resp_text, item_dict):
    """
    If 'skip' in the prompt => (None, ''), skipping the item.
    Otherwise => (1.0, ''), accepted with reward=1.0.
    """
    if "skip" in item_dict["prompt"].lower():
        return (None, "")
    return (1.0, "")

@pytest.fixture
def mock_teacher_model():
    """
    Minimal placeholder representing a teacher model.
    Could be a real model or just a string for testing.
    """
    return "mock_teacher_model"

@pytest.fixture
def mock_tokenizer():
    """
    Minimal tokenizer placeholder.
    """
    return "mock_tokenizer"

def test_collect_teacher_data_once_basic(mock_teacher_model, mock_tokenizer, caplog):
    """
    Tests 'collect_teacher_data_once' with a small dataset, 
    skipping items if 'skip' is in the prompt. 
    We pass device='torch' => framework='torch'.
    """
    dataset = [
        {"prompt": "Question1: do something."},
        {"prompt": "Question2: skip this one please."},
        {"prompt": "Question3: final question."},
    ]

    # We'll define a small local single-pass function 
    # or we can just call 'collect_teacher_data_once(...)' directly.

    final_data = collect_teacher_data_once(
        teacher_model=mock_teacher_model,
        tokenizer=mock_tokenizer,
        dataset=dataset,
        calculate_reward=calculate_reward_mock,
        batch_size=2,
        G=1,
        device="torch",  # => 'torch' logic
        verbose=True,
        generate_single_fn=generate_teacher_mock
    )

    # The second item has "skip" => reward=None => skip => 2 items remain
    assert len(final_data) == 2
    item1 = final_data[0]
    # For G=1 => 1 response, 1 logprob, 1 reward
    assert "responses" in item1 and len(item1["responses"]) == 1
    assert "teacher_logprobs" in item1 and len(item1["teacher_logprobs"]) == 1
    assert "rewards" in item1 and len(item1["rewards"]) == 1
    # logprob=1.23
    assert item1["teacher_logprobs"][0] == 1.23
    # reward=1.0
    assert item1["rewards"][0] == 1.0

    # Check skipping
    skipped_prompts = [x["item"]["prompt"] for x in final_data]
    assert all("skip" not in prompt.lower() for prompt in skipped_prompts)

def test_collect_teacher_data_once_empty(mock_teacher_model, mock_tokenizer):
    """
    Tests single-pass approach with empty dataset => no items collected.
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

    # Empty => no items
    assert len(final_data) == 0
