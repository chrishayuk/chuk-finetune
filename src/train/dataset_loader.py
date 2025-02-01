# src/train/dataset_loader.py
from typing import Any

def get_dataloader(framework: str, dataset: Any, batch_size: int, shuffle: bool = True):
    """
    Returns a dataset loader depending on the framework: 'torch' or 'mlx'.
    
    For Torch, returns a torch DataLoader. 
    For MLX, returns a naive list batching approach.
    """
    if framework == "torch":
        from torch.utils.data import DataLoader

        # return the data loader
        return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)

    elif framework == "mlx":
        # MLX doesn't have an equivalent built-in. 
        # We'll do a naive approach returning a list of batches:
        data = list(dataset)

        # Return a simple generator function
        def mlx_data_iterator():
            for i in range(0, len(data), batch_size):
                yield [str(q) for q in data[i : i + batch_size]]

        return mlx_data_iterator
    else:
        raise ValueError(f"Unknown framework: {framework}")
