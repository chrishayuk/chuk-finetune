# tests/train/grpo/test_grpo_prepare.py

import pytest
from unittest.mock import MagicMock

from train.grpo.grpo_prepare import prepare_batch_data_for_grpo

def test_prepare_batch_data_valid_items():
    """
    Test the normal path: items are valid, rewards are never None,
    G=2 => each item has 2 responses, old_logprobs, rewards.
    """
    # Mock ensure_dict_fn to return {"prompt": <some prompt>} for each item
    ensure_dict_fn = MagicMock(side_effect=lambda x: {"prompt": x})

    # Mock generate_single_fn => (response_text, sum_lp)
    # We'll just do f"resp_for_{prompt}", 0.123
    generate_single_fn = MagicMock(
        side_effect=lambda prompt, verbose: (f"resp_for_{prompt}", 0.123)
    )

    # Mock calculate_reward => (score, feedback_text)
    # We'll always return (1.0, "ok") for demonstration
    calculate_reward = MagicMock(return_value=(1.0, "ok"))

    batch_questions = ["Hello", "World"]  # 2 items
    G = 2

    results = prepare_batch_data_for_grpo(
        batch_questions=batch_questions,
        ensure_dict_fn=ensure_dict_fn,
        generate_single_fn=generate_single_fn,
        calculate_reward=calculate_reward,
        G=G,
        verbose=False
    )

    # We expect 2 items in results
    assert len(results) == 2

    for idx, res in enumerate(results):
        # "item" => should be {"prompt": <the raw item>}, from ensure_dict_fn
        assert res["item"] == {"prompt": batch_questions[idx]}

        # "responses" => G=2 strings, each "resp_for_..."
        assert len(res["responses"]) == G
        for r_txt in res["responses"]:
            assert r_txt == f"resp_for_{batch_questions[idx]}"

        # "old_logprobs" => G=2 floats
        assert len(res["old_logprobs"]) == G
        for lp in res["old_logprobs"]:
            assert lp == 0.123

        # "rewards" => G=2 floats
        assert len(res["rewards"]) == G
        for rw in res["rewards"]:
            assert rw == 1.0

    # Check how many times we called our mocks
    # ensure_dict_fn => once per item
    assert ensure_dict_fn.call_count == 2
    # generate_single_fn => called G times for each valid item => 2 * 2 = 4 calls
    assert generate_single_fn.call_count == 4
    # calculate_reward => same number => 4
    assert calculate_reward.call_count == 4


def test_prepare_batch_data_invalid_item():
    """
    If ensure_dict_fn returns None, item is skipped entirely.
    """
    # Make the first item "valid" => returns {"prompt": "Item1"},
    # and the second item "invalid" => returns None.
    ensure_dict_fn = MagicMock(side_effect=lambda x: {"prompt": x} if x == "Item1" else None)
    generate_single_fn = MagicMock(return_value=("mock_resp", 0.0))
    calculate_reward = MagicMock(return_value=(1.0, "good"))

    batch_questions = ["Item1", "Item2"]

    results = prepare_batch_data_for_grpo(
        batch_questions=batch_questions,
        ensure_dict_fn=ensure_dict_fn,
        generate_single_fn=generate_single_fn,
        calculate_reward=calculate_reward,
        G=2,
        verbose=False
    )

    # We should only have 1 item in the result (Item1).
    assert len(results) == 1
    # Since we returned {"prompt": "Item1"} for the first item, the pipeline
    # records that as the 'item' field.
    assert results[0]["item"] == {"prompt": "Item1"}

    # The second item was invalid => not processed => no calls to generate_single_fn for it
    # Check call counts
    assert ensure_dict_fn.call_count == 2  # called for both items
    # For item1, G=2 => 2 calls to generate_single_fn
    assert generate_single_fn.call_count == 2
    # Similarly, 2 calls to calculate_reward
    assert calculate_reward.call_count == 2



def test_prepare_batch_data_reward_is_none():
    """
    If 'calculate_reward' returns None for ANY response, skip the entire item.
    We'll simulate that the second response for item1 yields None => skip item1 entirely.
    """
    ensure_dict_fn = MagicMock(return_value={"prompt": "Test"})
    generate_single_fn = MagicMock(side_effect=[
        ("resp_for_Test_0", 0.0),
        ("resp_for_Test_1", 0.1),
    ])

    # We'll return (1.0, "ok") for the first call, (None, "fail") for the second
    calculate_reward = MagicMock(side_effect=[
        (1.0, "ok"),
        (None, "fail")
    ])

    batch_questions = ["Q1"]
    results = prepare_batch_data_for_grpo(
        batch_questions=batch_questions,
        ensure_dict_fn=ensure_dict_fn,
        generate_single_fn=generate_single_fn,
        calculate_reward=calculate_reward,
        G=2,
        verbose=False
    )

    # Because the second call returns None => we skip the entire item => no results
    assert len(results) == 0

    # Check calls
    ensure_dict_fn.assert_called_once()
    # generate_single_fn => called only 2 times for item1 (but we skip after 2nd anyway)
    assert generate_single_fn.call_count == 2
    # calculate_reward => also 2 calls (the second returned None)
    assert calculate_reward.call_count == 2


def test_prepare_batch_data_empty_input():
    """
    An empty list => returns an empty list, no calls to anything.
    """
    ensure_dict_fn = MagicMock()
    generate_single_fn = MagicMock()
    calculate_reward = MagicMock()

    batch_questions = []

    results = prepare_batch_data_for_grpo(
        batch_questions=batch_questions,
        ensure_dict_fn=ensure_dict_fn,
        generate_single_fn=generate_single_fn,
        calculate_reward=calculate_reward,
        G=2,
        verbose=False
    )

    assert results == []
    ensure_dict_fn.assert_not_called()
    generate_single_fn.assert_not_called()
    calculate_reward.assert_not_called()


@pytest.mark.parametrize("G", [1, 3])
def test_prepare_batch_data_varying_G(G):
    """
    Quick check that if G changes, we get that many responses per item.
    """
    ensure_dict_fn = MagicMock(side_effect=lambda x: {"prompt": x})
    generate_single_fn = MagicMock(return_value=("mock_resp", 0.0))
    calculate_reward = MagicMock(return_value=(10.0, "feedback"))
    batch_questions = ["Q1", "Q2"]

    results = prepare_batch_data_for_grpo(
        batch_questions=batch_questions,
        ensure_dict_fn=ensure_dict_fn,
        generate_single_fn=generate_single_fn,
        calculate_reward=calculate_reward,
        G=G,
        verbose=True  # can test 'verbose' doesn't break
    )

    # Expect length == 2
    assert len(results) == 2
    for res in results:
        assert len(res["responses"]) == G
        assert len(res["old_logprobs"]) == G
        assert len(res["rewards"]) == G

    # We can also check call counts: G calls per item => 2 * G
    assert generate_single_fn.call_count == 2 * G
    assert calculate_reward.call_count == 2 * G
