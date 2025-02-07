# src/train/dataset_loader.py

import json
import random
from typing import Any, List, Callable, Union, Optional, Iterable

def load_jsonl_as_list(jsonl_path: str) -> List[dict]:
    """
    Reads a JSON Lines file, returning a list of dicts.
    
    Each line in the file must be a complete JSON object, for example:
      {"prompt": "Work out 62 - 97", "verifiers": [...]}
    Blank lines are skipped. 
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


def collate_as_list_of_dicts(batch: List[Any]) -> List[Any]:
    """
    A trivial collate function for PyTorch that returns the batch
    as a Python list of items (dicts, strings, etc.) without converting
    them into tensors. This is useful if your items are not homogeneous
    or you want to handle them manually in 'train_step'.
    """
    # You could do transformations here if needed, but the default is just pass-through.
    return list(batch)


def get_dataloader(
    framework: str,
    dataset: Union[str, List[Any]],
    batch_size: int,
    shuffle: bool = True,
    torch_collate_fn: Optional[Callable] = collate_as_list_of_dicts,
) -> Callable[[Optional[int]], Iterable[Any]]:
    """
    Returns a callable that, when invoked (optionally with a custom batch_size),
    yields mini-batches of data.

    Args:
        framework (str):
            One of {"torch", "mlx"}. Determines the iteration logic.
        dataset (str | list):
            - If a string, assumed to be a path to a JSONL file, which we load in-memory.
            - If a list (or something list-like), we use it directly.
            - (Advanced) If it's already a PyTorch Dataset object, we pass it directly to DataLoader in the Torch case.
        batch_size (int):
            The default mini-batch size (can be overridden at call time).
        shuffle (bool):
            If True, data is shuffled each time we create a new data iterator:
              - Torch => pass `shuffle=True` to the DataLoader
              - MLX => use `random.shuffle` on a copy of the data
        torch_collate_fn (Callable, optional):
            A collate function to use in the PyTorch DataLoader. 
            Defaults to `collate_as_list_of_dicts`, 
            which returns raw lists of items without tensor conversion. 
            If you want PyTorch's default stacking, set this to None.

    Returns:
        A function that can be called like `data_iterator_fn(batch_size_override)`
        to produce an iterator (generator) over batches.

    Example:
        >>> data_iterator_fn = get_dataloader("torch", "mydata.jsonl", batch_size=4, shuffle=True)
        >>> for epoch in range(3):
        ...     for batch in data_iterator_fn():
        ...         # process batch items
    """

    # 1) If dataset is a string, assume it's a path to a JSONL file => load it
    if isinstance(dataset, str):
        dataset_list = load_jsonl_as_list(dataset)
    else:
        # If it's already a list or a torch Dataset object, keep it as is;
        # if it's a list-like, we do `list(dataset)` to fix iteration
        dataset_list = dataset

    # 2) Build a data iterator function depending on framework
    if framework == "torch":
        from torch.utils.data import DataLoader, Dataset

        def torch_data_iterator(batch_size_override: Optional[int] = None):
            current_bsz = batch_size_override or batch_size

            if isinstance(dataset_list, list):
                # If it's a raw Python list, wrap it in a simple Dataset for the DataLoader
                class ListDataset(Dataset):
                    def __init__(self, data_list: List[Any]):
                        self.data_list = data_list

                    def __len__(self):
                        return len(self.data_list)

                    def __getitem__(self, idx):
                        return self.data_list[idx]

                ds = ListDataset(dataset_list)
            else:
                # Otherwise, assume it's already a valid Torch Dataset
                ds = dataset_list

            loader = DataLoader(
                ds,
                batch_size=current_bsz,
                shuffle=shuffle,
                collate_fn=torch_collate_fn
            )
            for batch in loader:
                yield batch

        return torch_data_iterator

    elif framework == "mlx":
        # Convert to a list if it's not
        if not isinstance(dataset_list, list):
            dataset_list = list(dataset_list)

        def mlx_data_iterator(batch_size_override: Optional[int] = None):
            current_bsz = batch_size_override or batch_size

            # Shuffle each time we create a new iterator, 
            # so calling it for multiple epochs yields different orders
            data_copy = dataset_list[:]  # shallow copy
            if shuffle:
                random.shuffle(data_copy)

            for i in range(0, len(data_copy), current_bsz):
                yield data_copy[i : i + current_bsz]

        return mlx_data_iterator

    else:
        raise ValueError(f"Unknown framework: {framework}")
