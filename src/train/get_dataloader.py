# src/train/dataset_loader.py
from typing import Any, List, Union
import random
import json
import os

def load_dataset_any_format(json_path: str) -> List[dict]:
    """
    Tries to load a file that could be:
      1) A single JSON object (possibly multi-line).
         e.g.,
         {
           "prompt": "Work out 62 - 97 and display the answer",
           "verifiers": [...]
         }
      2) A JSON array of objects (multi-line or single line).
      3) A JSON Lines file (.jsonl) with one object per line.

    We'll try to parse the entire file with json.load(...) first. If that fails,
    we fall back to line-by-line parsing. The result is always a list of dicts.
    """
    # 1) Try single-shot parsing (handles single-object or array)
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        # If data is a dict, wrap in a list so we return List[dict]
        if isinstance(data, dict):
            return [data]
        # If it's already a list, assume it's a list of dicts
        elif isinstance(data, list):
            # Optionally check that each element is a dict, if you want
            return data
        else:
            raise ValueError(
                f"File {json_path} loaded, but top-level JSON is neither an object nor an array."
            )
    except json.JSONDecodeError:
        pass  # Fallback to line-by-line approach

    # 2) Fall back to line-by-line "JSONL" parsing
    data_list = []
    with open(json_path, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip blank lines
            record = json.loads(line)
            if not isinstance(record, dict):
                raise ValueError(
                    f"Line in {json_path} is not a JSON object:\n{line}"
                )
            data_list.append(record)
    return data_list


def get_dataloader(
    framework: str,
    dataset: Any,       # can be a path string, a Python list, or a torch Dataset
    batch_size: int,
    shuffle: bool = True
):
    """
    Returns a data iterator function (callable) that can be used as:
        data_iterator_fn(batch_size_override=None)

    If `dataset` is a string, we assume it's a path to a JSON or JSONL file
    and try to parse it with load_dataset_any_format(...).
    Otherwise, if `dataset` is already a list or a Torch Dataset, we use it directly.

    For framework='torch', we wrap the result in a DataLoader.
    For framework='mlx', we do manual batching.
    """

    # 1) If dataset is a string, assume it's a file path
    if isinstance(dataset, str) and os.path.isfile(dataset):
        dataset_list = load_dataset_any_format(dataset)
        dataset = dataset_list  # Overwrite `dataset` with the in-memory list

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
