# src/train/dataset_loader.py
from typing import Any
import random

def get_dataloader(framework: str, dataset: Any, batch_size: int, shuffle: bool = True):
    """
    Returns a data iterator function that can be called as:
        data_iterator_fn(batch_size_override)
    If batch_size_override is None, it defaults to the original batch_size.
    """

    if framework == "torch":
        from torch.utils.data import DataLoader

        def torch_data_iterator(batch_size_override=None):
            """
            Creates and yields batches from a PyTorch DataLoader. 
            Allows overriding the batch size at call time.
            """
            # Fallback to the original batch_size if none provided
            current_bsz = batch_size_override or batch_size

            # Construct the DataLoader with the current batch size
            loader = DataLoader(dataset, batch_size=current_bsz, shuffle=shuffle)
            for batch in loader:
                yield batch

        return torch_data_iterator

    elif framework == "mlx":
        # Convert to list (if not already) to allow manual batching
        data_list = list(dataset)

        # Shuffle if specified
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