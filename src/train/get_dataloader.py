# src/train/dataset_loader.py
from typing import Any, List
import random
import json


def load_jsonl_as_list(jsonl_path: str) -> List[dict]:
    """
    Reads a JSON Lines file, returning a list of dicts.
    Each line must be a complete JSON object:
      {
        "prompt": "...",
        "verifiers": [...]
      }
    or similar.

    Example line in the file:
    {"prompt": "Work out 62 - 97", "verifiers": [...]}
    """
    data_list = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip blank lines
            record = json.loads(line)  # parse the entire line as JSON
            data_list.append(record)
    return data_list


def get_dataloader(
    framework: str,
    dataset: Any,       # can be a path string or an in-memory list
    batch_size: int,
    shuffle: bool = True
):
    """
    Returns a data iterator function that can be called as:
        data_iterator_fn(batch_size_override)

    If dataset is a string, we assume it's a path to a .jsonl file and we load it.
    Otherwise, if dataset is already a list/torch Dataset, we use it directly.
    """

    # 1) If dataset is a string, assume it's a path to a JSONL file
    if isinstance(dataset, str):
        dataset_path = dataset
        data_list = load_jsonl_as_list(dataset_path)
        dataset = data_list  # Overwrite `dataset` with the in-memory list

    # 2) Create the iterator based on the framework
    if framework == "torch":
        from torch.utils.data import DataLoader

        def torch_data_iterator(batch_size_override=None):
            """
            Creates and yields batches from a PyTorch DataLoader. 
            Allows overriding the batch size at call time.
            """
            current_bsz = batch_size_override or batch_size
            loader = DataLoader(dataset, batch_size=current_bsz, shuffle=shuffle)
            for batch in loader:
                yield batch

        return torch_data_iterator

    elif framework == "mlx":
        # Convert to list (if not already) to allow manual batching
        data_list = list(dataset)

        if shuffle:
            random.shuffle(data_list)

        def mlx_data_iterator(batch_size_override=None):
            """
            Manually yields slices of data_list in mini-batches.
            Allows overriding the batch size at call time.
            """
            current_bsz = batch_size_override or batch_size
            for i in range(0, len(data_list), current_bsz):
                yield data_list[i : i + current_bsz]

        return mlx_data_iterator

    else:
        raise ValueError(f"Unknown framework: {framework}")
