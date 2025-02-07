# src/train/dataset_loader.py
import json
import random
from typing import Any, List, Callable, Union, Optional, Iterable

def load_jsonl_as_list(jsonl_path: str) -> List[dict]:
    """
    Reads a JSON Lines file, returning a list of dicts.
    Each line must be a complete JSON object, e.g.:
        {
          "prompt": "...",
          "verifiers": [...]
        }

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

def collate_as_list_of_dicts(batch: List[Any]) -> List[Any]:
    """
    A trivial collate function for PyTorch that returns the batch
    as a Python list of items (dicts, strings, etc.) without converting
    them into tensors. This is useful if your items are not homogeneous
    or you want to handle them manually in 'train_step'.
    """
    # Optionally, you could do transformations here if needed
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
            - If a list, or another Python iterable, we use it directly.
            - (Advanced) If a torch Dataset object, we can also pass it directly in the Torch case.
        batch_size (int):
            The default mini-batch size.
        shuffle (bool):
            If True, data is shuffled (for Torch via DataLoader shuffle, for MLX via random.shuffle).
        torch_collate_fn (Callable, optional):
            A collate function to use in the PyTorch DataLoader. Defaults to `collate_as_list_of_dicts`,
            which returns raw lists of items. If you want PyTorch's default stacking, you can set this to None.

    Returns:
        A function that can be called like `data_iterator_fn(batch_size_override)` to produce an iterator over batches.

    Example:
        >>> data_iterator_fn = get_dataloader("torch", "mydata.jsonl", batch_size=4, shuffle=True)
        >>> for epoch in range(3):
        ...     for batch in data_iterator_fn():
        ...         # process batch
    """

    # 1) If dataset is a string, assume it's a path to a JSONL file => load it
    if isinstance(dataset, str):
        dataset_path = dataset
        dataset_list = load_jsonl_as_list(dataset_path)
    else:
        # If it's already a list or something else, just keep it as is
        # If it's a torch Dataset object, we pass it to DataLoader below
        dataset_list = dataset

    # 2) Build a data iterator function depending on framework
    if framework == "torch":
        # We can import Torch here so that it's only needed if we actually use the Torch framework
        from torch.utils.data import DataLoader, Dataset

        def torch_data_iterator(batch_size_override: Optional[int] = None):
            current_bsz = batch_size_override or batch_size

            if isinstance(dataset_list, list):
                # If it's a raw Python list, we can wrap it in a simple Dataset for DataLoader
                class ListDataset(Dataset):
                    def __init__(self, data_list):
                        self.data_list = data_list
                    def __len__(self):
                        return len(self.data_list)
                    def __getitem__(self, idx):
                        return self.data_list[idx]

                ds = ListDataset(dataset_list)
            else:
                # Otherwise assume dataset_list is already a Torch Dataset
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
        # If shuffle, we do a random shuffle once each time the iterator is called
        # That means each call to data_iterator_fn() is effectively a new epoch
        if not isinstance(dataset_list, list):
            # Convert to a list if it's not one
            dataset_list = list(dataset_list)

        def mlx_data_iterator(batch_size_override: Optional[int] = None):
            current_bsz = batch_size_override or batch_size

            # Shuffle the data for each new iteration if requested
            data_copy = dataset_list[:]  # copy so as not to mutate original
            if shuffle:
                random.shuffle(data_copy)

            # Yield slices
            for i in range(0, len(data_copy), current_bsz):
                yield data_copy[i : i + current_bsz]

        return mlx_data_iterator

    else:
        raise ValueError(f"Unknown framework: {framework}")
