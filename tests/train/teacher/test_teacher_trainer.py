# tests/test_teacher_trainer.py
import pytest
from train.teacher.teacher_trainer import TeacherTrainer

def generate_teacher_mock(prompt, verbose=False):
    """
    Mock teacher generation: returns prompt + ' => teacher_out' and logprob=1.23.
    """
    response_text = prompt + " => teacher_out"
    logprob = 1.23
    return (response_text, logprob)

def calculate_reward_mock(response_text, item_dict):
    """
    If 'skip' in the prompt => (None, ""), skipping the item.
    Otherwise => (1.0, ""), meaning accepted with reward=1.0.
    """
    if "skip" in item_dict["prompt"].lower():
        return (None, "")
    return (1.0, "")

@pytest.fixture
def teacher_trainer():
    """
    A fixture returning a TeacherTrainer instance with G=2,
    using the mock generation & reward functions.
    """
    teacher_model = "mock_teacher_model"
    tokenizer = "mock_tokenizer"

    # No 'ensure_dict_fn' => we rely on inlined logic in prepare_batch_data_for_teacher 
    # that checks if item is a dict with 'prompt'.
    trainer = TeacherTrainer(
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        calculate_reward=calculate_reward_mock,
        generate_single_fn=generate_teacher_mock,
        G=2,
        device=None,
        verbose=False
    )
    return trainer

def test_teacher_trainer_basic(teacher_trainer):
    """
    Tests that prepare_batch_data => train_step produces expected structure,
    skipping items if 'skip' is in the prompt.
    """
    # A small dataset
    dataset = [
        {"prompt": "Q1: compute 2+2?"},
        {"prompt": "Q2: please skip me."},
        {"prompt": "Q3: write a short poem."}
    ]

    # We do a single batch => call prepare_batch_data
    batch_data = teacher_trainer.prepare_batch_data(dataset)
    # The second item has 'skip' => skip => only 2 remain
    assert len(batch_data) == 2, "Expected 2 items after skipping"

    # Check each item
    item1 = batch_data[0]
    item2 = batch_data[1]

    # Since G=2 => each item has 2 responses, 2 teacher_logprobs, 2 rewards
    assert "responses" in item1 and len(item1["responses"]) == 2
    assert "teacher_logprobs" in item1 and len(item1["teacher_logprobs"]) == 2
    assert "rewards" in item1 and len(item1["rewards"]) == 2

    # Check appended text
    assert " => teacher_out" in item1["responses"][0]

    # Now call train_step => returns (mean_reward, batch_data)
    mean_reward, processed_data = teacher_trainer.train_step(batch_data)
    # We expect the same data
    assert processed_data == batch_data, "train_step should return the same data"
    # We have 2 items * 2 responses => 4 total responses => each reward=1.0 => mean_reward=1.0
    assert pytest.approx(mean_reward, 0.001) == 1.0

def test_teacher_trainer_no_data(teacher_trainer):
    """
    Tests passing an empty dataset => no items => train_step => (0.0, []).
    """
    empty_batch = []
    batch_data = teacher_trainer.prepare_batch_data(empty_batch)
    assert len(batch_data) == 0

    mean_reward, processed_data = teacher_trainer.train_step(batch_data)
    assert mean_reward == 0.0
    assert processed_data == batch_data == []
