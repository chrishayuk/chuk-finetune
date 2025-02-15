import logging
import sys

# logging
logger = logging.getLogger(__name__)

def debug_tensor_info(tensor, name: str):
    """
    Logs the shape, data type, and reference count of a tensor.
    
    Args:
        tensor: The tensor to inspect.
        name (str): A label for the tensor.
    """
    if tensor is None:
        logger.debug(f"{name} is None")
    else:
        # sys.getrefcount returns the number of references including the temporary reference in the call.
        ref_count = sys.getrefcount(tensor)
        logger.debug(f"{name} - shape: {tensor.shape}, dtype: {tensor.dtype}, ref_count: {ref_count}")