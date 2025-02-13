import pytest
from train.teacher.teacher_trainer import TeacherTrainer

def generate_teacher_mock(teacher_model, tokenizer, prompt, verbose=False, max_new_tokens=1024, temperature=0.7, top_p=0.95):
    """
    Mock teacher generation: returns (prompt + ' => teacher_out', logprob=1.23).
    Signature matches the unified teacher generation function.
    """
    response_text = prompt + " => teacher_out"
    return (response_text, 1.23)

def calculate_reward_mock(response_text, item_dict):
    """
    If 'skip' is in the prompt => returns (None, ""), causing the item to be skipped.
    Otherwise, returns (1.0, ""), accepted with reward 1.0.
    """
    if "skip" in item_dict["prompt"].lower():
        return (None, "")
    return (1.0, "")

@pytest.fixture
def teacher_trainer():
    """
    A fixture returning a TeacherTrainer instance with G=2,
    using the updated mock generation and reward functions.
    """
    teacher_model = "mock_teacher_model"
    tokenizer = "mock_tokenizer"
    # No ensure_dict_fn is passed.
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
    Tests that prepare_batch_data and train_step produce the expected structure,
    skipping items if 'skip' is in the prompt.
    """
    # Define a small dataset.
    dataset = [
        {"prompt": "Q1: compute 2+2?"},
        {"prompt": "Q2: please skip me."},
        {"prompt": "Q3: write a short poem."}
    ]

    # Call prepare_batch_data to process the dataset.
    batch_data = teacher_trainer.prepare_batch_data(dataset)
    # The second item should be skipped => we expect 2 items remain.
    assert len(batch_data) == 2, "Expected 2 items after skipping"

    # Check the structure of the first item.
    item1 = batch_data[0]
    assert "responses" in item1 and len(item1["responses"]) == 2
    assert "teacher_logprobs" in item1 and len(item1["teacher_logprobs"]) == 2
    assert "rewards" in item1 and len(item1["rewards"]) == 2
    # Verify that the generated response text ends with " => teacher_out".
    assert " => teacher_out" in item1["responses"][0]

    # Call train_step and verify that the mean reward is computed correctly.
    mean_reward, processed_data = teacher_trainer.train_step(batch_data)
    assert processed_data == batch_data, "train_step should return the same data"
    # Since there are 2 items * 2 responses each and each reward is 1.0, mean_reward should be 1.0.
    assert pytest.approx(mean_reward, 0.001) == 1.0

def test_teacher_trainer_no_data(teacher_trainer):
    """
    Tests that passing an empty dataset returns an empty batch_data,
    and that train_step returns (0.0, []).
    """
    empty_batch = []
    batch_data = teacher_trainer.prepare_batch_data(empty_batch)
    assert len(batch_data) == 0

    mean_reward, processed_data = teacher_trainer.train_step(batch_data)
    assert mean_reward == 0.0
    assert processed_data == batch_data == []
