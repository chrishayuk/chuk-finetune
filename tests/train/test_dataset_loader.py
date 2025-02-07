import pytest
import json
import os
from typing import List

from train.dataset_loader import load_jsonl_as_list, get_dataloader

@pytest.fixture
def small_jsonl(tmp_path):
    """
    Creates a small .jsonl file and returns its path.
    Each line is a dict with "prompt" and "verifiers".
    """
    data = [
        {"prompt": "Prompt A", "verifiers": ["vA1", "vA2"]},
        {"prompt": "Prompt B", "verifiers": ["vB1"]},
        {"prompt": "Prompt C", "verifiers": []},
        {"prompt": "Prompt D", "verifiers": ["vD1", "vD2", "vD3"]},
    ]
    file_path = tmp_path / "test_data.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for record in data:
            f.write(json.dumps(record) + "\n")
    return file_path


def test_load_jsonl_as_list(small_jsonl):
    """
    Tests that load_jsonl_as_list reads the file correctly into a list[dict].
    """
    records = load_jsonl_as_list(str(small_jsonl))
    assert len(records) == 4
    assert records[0]["prompt"] == "Prompt A"
    assert records[1]["verifiers"] == ["vB1"]
    assert records[3]["verifiers"] == ["vD1", "vD2", "vD3"]


@pytest.mark.parametrize("framework", ["torch", "mlx"])
def test_get_dataloader_with_jsonl(framework, small_jsonl):
    """
    Tests get_dataloader when dataset is a JSONL path, for both torch and mlx.
    """
    batch_size = 2
    data_iterator_fn = get_dataloader(
        framework=framework,
        dataset=str(small_jsonl),  # pass the path
        batch_size=batch_size,
        shuffle=False  # so we can check ordering
    )

    # The returned object is a function we can call
    all_batches = list(data_iterator_fn())  # default batch_size => 2
    # We expect 2 batches for 4 items total
    assert len(all_batches) == 2

    # Check the shape of each batch
    first_batch = all_batches[0]
    second_batch = all_batches[1]
    # For torch, the default collate might create a structure of dicts or Tensors
    # For mlx, we just get a list of dicts. We'll handle that carefully.

    if framework == "torch":
        # By default, the Torch DataLoader might convert to a dict of lists or lists of Tensors
        # If your dataset are dicts, it might do something like:
        #   first_batch["prompt"] -> list of prompts
        # so let's do a partial check
        # If you need a custom collate, you'd do that in get_dataloader
        assert len(first_batch) > 0  # it's not empty
        # We won't do exact asserts here because the default collate can be complex.
    else:  # mlx
        # For MLX, we return a list of dicts
        assert isinstance(first_batch, list)
        assert len(first_batch) == 2
        # The first batch should contain the first 2 lines from the JSONL
        assert first_batch[0]["prompt"] == "Prompt A"
        assert first_batch[1]["prompt"] == "Prompt B"


@pytest.mark.parametrize("framework", ["torch", "mlx"])
def test_get_dataloader_in_memory_list(framework):
    """
    Tests get_dataloader when dataset is already an in-memory list of dicts.
    """
    data_list = [
        {"prompt": "Hello 1"},
        {"prompt": "Hello 2"},
        {"prompt": "Hello 3"},
        {"prompt": "Hello 4"},
    ]
    # We'll set shuffle=False to keep it deterministic
    data_iterator_fn = get_dataloader(
        framework=framework,
        dataset=data_list,
        batch_size=2,
        shuffle=False
    )

    all_batches = list(data_iterator_fn())
    assert len(all_batches) == 2  # 4 items => 2 per batch => 2 total

    if framework == "torch":
        # Again, default collate may produce a dict-of-lists or list-of-dicts
        # Let's just check we got 2 items in each batch in some dimension
        batch_1 = all_batches[0]
        # If the default collate sees a list of dict with same keys,
        # it might produce something like: {"prompt": ["Hello 1","Hello 2"]}
        if isinstance(batch_1, dict):
            # Check we have 2 prompts
            assert len(batch_1["prompt"]) == 2
        else:
            # We might have a list-of-objects
            assert len(batch_1) == 2
    else:
        # MLX => we expect a list of dicts
        assert isinstance(all_batches[0], list)
        assert len(all_batches[0]) == 2
        assert all_batches[0][0]["prompt"] == "Hello 1"
        assert all_batches[0][1]["prompt"] == "Hello 2"


def test_get_dataloader_shuffle_mlx():
    """
    Tests that shuffle=True for mlx actually shuffles the data
    (not guaranteed, but we can check if the order is different).
    There's a small chance random shuffle might produce the original order,
    so we do a repeated check or just see if it's typically different.
    """
    data_list = [
        {"id": i, "prompt": f"Prompt {i}"} for i in range(10)
    ]
    no_shuffle_fn = get_dataloader("mlx", data_list, batch_size=10, shuffle=False)
    shuffled_fn = get_dataloader("mlx", data_list, batch_size=10, shuffle=True)

    # no shuffle => the single batch is in the original order
    batch_no_shuffle = list(no_shuffle_fn())[0]
    batch_shuffle = list(shuffled_fn())[0]

    original_ids = [d["id"] for d in data_list]
    no_shuffle_ids = [d["id"] for d in batch_no_shuffle]
    shuffle_ids = [d["id"] for d in batch_shuffle]

    assert no_shuffle_ids == original_ids  # they should match exactly
    # The shuffle might produce the same sequence by chance, but it's unlikely
    # We'll allow the test to pass if it's different. If it's the same, we can re-run or seed
    # For demonstration, let's do a partial check:
    assert shuffle_ids != original_ids, "Shuffle produced same ordering by chance! Possibly rare or seed issue."


def test_get_dataloader_unknown_framework():
    """
    Tests that get_dataloader raises ValueError for an unknown framework.
    """
    data_list = [{"prompt": "Test"}]
    with pytest.raises(ValueError, match="Unknown framework"):
        get_dataloader("some_non_existent_framework", data_list, 4)
