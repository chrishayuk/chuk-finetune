# src/train/dataset_loader.py
from typing import Any
import random

def get_dataloader(framework: str, dataset: Any, batch_size: int, shuffle: bool = True):
    if framework == "torch":
        from torch.utils.data import DataLoader
        
        # We'll wrap the torch DataLoader in a zero-argument function
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

        def torch_data_iterator():
            for batch in loader:
                yield batch

        return torch_data_iterator

    elif framework == "mlx":
        data = list(dataset)
        if shuffle:
            random.shuffle(data)

        # Return a zero-argument function for MLX
        def mlx_data_iterator():
            for i in range(0, len(data), batch_size):
                yield data[i : i + batch_size]
        return mlx_data_iterator

    else:
        raise ValueError(f"Unknown framework: {framework}")