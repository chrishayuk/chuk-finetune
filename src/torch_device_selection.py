# src/torch_device_selection.py
import torch

class DeviceSelector:
    """
    A utility class for selecting a device (CPU, CUDA, or MPS).
    If a forced device is specified but not available, it falls back
    according to the logic below.
    """

    @staticmethod
    def get_preferred_device(forced_device: str = None) -> torch.device:
        """
        Returns a torch.device object based on user preference and availability.

        1. If 'forced_device' is provided, attempt to use it.
           If it's not available, fall back to the best available.
        2. If 'forced_device' is not provided, prefer CUDA, then MPS, else CPU.
        """
        # If the user specified a device, check availability
        if forced_device is not None:
            device_lower = forced_device.lower()

            # check if cuda is available
            if device_lower == "cuda" and torch.cuda.is_available():
                # use cuda
                return torch.device("cuda")
            # check if mps is available
            elif device_lower == "mps" and getattr(torch.backends.mps, "is_available", lambda: False)():
                # use mps
                return torch.device("mps")
            # use cpu
            elif device_lower == "cpu":
                # use cpu
                return torch.device("cpu")
            
            # If forced device isn't available, we'll fall back
            print(f"Warning: Forced device '{forced_device}' not available, falling back to auto-detect.")

        # Auto-detect best available device
        if torch.cuda.is_available():
            # use cuda
            return torch.device("cuda")
        elif getattr(torch.backends.mps, "is_available", lambda: False)():
            # use mps
            return torch.device("mps")
        else:
            # use cpu
            return torch.device("cpu")
