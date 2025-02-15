# src/device/torch_device_memory.py
import logging
import torch

# logging
logger = logging.getLogger(__name__)

def log_device_memory(device: str = None, tag: str = ""):
    """
    Logs memory usage for CUDA and MPS devices.

    For CUDA:
      - Allocated memory and reserved memory (in MB)

    For MPS:
      - Current allocated memory, driver allocated memory, and the recommended max memory (in MB)

    Args:
        device (str): "cuda", "mps", or "cpu". If None, auto-detects available device.
        tag (str): A label to include in the log message.
    """
    # Auto-detect device if not provided:
    if device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"

    if device == "cuda":
        allocated = torch.cuda.memory_allocated()
        reserved = torch.cuda.memory_reserved()
        print(f"[{tag}] CUDA Memory -> Allocated: {allocated / (1024**2):.2f} MB, Reserved: {reserved / (1024**2):.2f} MB")
    elif device == "mps":
        try:
            current_allocated = torch.mps.current_allocated_memory()  # in bytes
            driver_allocated = torch.mps.driver_allocated_memory()    # in bytes
            recommended_max = torch.mps.recommended_max_memory()      # in bytes

            logger.debug(f"[{tag}] MPS Memory -> Current Allocated: {current_allocated / (1024**2):.2f} MB, "
                  f"Driver Allocated: {driver_allocated / (1024**2):.2f} MB, "
                  f"Recommended Max: {recommended_max / (1024**2):.2f} MB")
        except Exception as e:
            logger.debug(f"[{tag}] Error reading MPS memory info: {e}")
    else:
        logger.debug(f"[{tag}] Device '{device}' does not support detailed memory tracking via this helper.")

# Example usage:
log_device_memory(tag="Before Forward Pass")
