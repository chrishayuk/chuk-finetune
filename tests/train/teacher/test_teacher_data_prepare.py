# tests/test_teacher_data_prepare.py

import pytest
from train.teacher.teacher_data_prepare import prepare_batch_data_for_teacher

def generate_single_test(prompt, verbose=False):
    """
    A mock teacher generation function.
    Returns (response_text, sum_logprob).
    Here we just append " -> teacher_response" for demonstration.
    sum_logprob = 1.23 as a dummy value.
    """
    response_text = prompt + " -> teacher_response"
    sum_lp = 1.23
    return (response_text, sum_lp)

def calculate_reward_always_ok(resp_text, item):
    """
    Always return a positive reward, never skip.
    """
    return (1.0, "")

def calculate_reward_sometimes_none(resp_text, item):
    """
    If 'prompt' contains the word 'skipme', return None => skip item.
    Otherwise return (1.0, "").
    """
    if "skipme" in item["prompt"].lower():
        return (None, "")
    return (1.0, "")

def test_prepare_batch_data_for_teacher_normal():
    """
    Test normal usage where all items produce valid teacher responses 
    and pass the reward check (none are skipped).
    """
    # Sample data
    batch_items = [
        {"prompt": "Question 1?"},
        {"prompt": "Question 2?"}
    ]

    # G=2 => generate 2 responses per item
    result = prepare_batch_data_for_teacher(
        batch_questions=batch_items,
        generate_single_fn=generate_single_test,
        calculate_reward=calculate_reward_always_ok,
        G=2,
        verbose=False
    )

    # We expect 2 items in the result
    assert len(result) == 2

    # Check the structure of the first item
    first_item = result[0]
    assert "item" in first_item
    assert "responses" in first_item
    assert "teacher_logprobs" in first_item
    assert "rewards" in first_item

    # We expect G=2 responses
    assert len(first_item["responses"]) == 2
    assert len(first_item["teacher_logprobs"]) == 2
    assert len(first_item["rewards"]) == 2

    # Check that the generation logic appended " -> teacher_response"
    assert " -> teacher_response" in first_item["responses"][0]

    # Check the sum_logprob is 1.23 from our mock
    assert first_item["teacher_logprobs"][0] == 1.23

    # Check the reward is 1.0
    assert first_item["rewards"][0] == 1.0

def test_prepare_batch_data_for_teacher_skips():
    """
    Test scenario where some items are skipped if reward=None.
    Also let's add an item missing 'prompt' to ensure it's skipped too.
    """
    batch_items = [
        {"prompt": "Question that is fine."},
        {"prompt": "Question that skipme?"},
        {"prompt": "Another question okay."},
        {"no_prompt_here": "invalid item => skip inline"}  # missing 'prompt'
    ]

    # We'll set G=1 for simplicity
    result = prepare_batch_data_for_teacher(
        batch_questions=batch_items,
        generate_single_fn=generate_single_test,
        calculate_reward=calculate_reward_sometimes_none,
        G=1,
        verbose=False
    )

    # Explanation of skipping:
    # 1) "Question that is fine." => valid, no 'skipme' => accepted
    # 2) "Question that skipme?" => has 'skipme' => reward=None => skip
    # 3) "Another question okay." => valid => accepted
    # 4) {"no_prompt_here": "invalid item => skip inline"} => missing 'prompt' => skip inline

    # So we expect 2 items in the final result
    assert len(result) == 2

    prompts_in_result = [r["item"]["prompt"] for r in result]
    # The second + fourth are skipped => we confirm only #1, #3 remain
    assert "Question that skipme?" not in prompts_in_result
    # The missing-'prompt' item won't appear, so no error
    assert "Question that is fine." in prompts_in_result
    assert "Another question okay." in prompts_in_result
