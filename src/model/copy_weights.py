# src/model/copy_weights.py

import logging
from model.model_detection import is_torch_model, is_mlx_model

logger = logging.getLogger(__name__)

def copy_weights(src_model, dst_model, strict: bool = True):
    """
    Copy parameters from 'src_model' to 'dst_model'.
    Both models must be recognized as either Torch or MLX.
    
    :param src_model: The reference model to copy from.
    :param dst_model: The model to copy into.
    :param strict: If True, raises an error if any parameters are missing or unmatched.
                   If False, attempts a "best effort" copy, ignoring missing keys in 
                   either source or destination.
    """
    src_is_torch = is_torch_model(src_model)
    dst_is_torch = is_torch_model(dst_model)
    src_is_mlx   = is_mlx_model(src_model)
    dst_is_mlx   = is_mlx_model(dst_model)

    if src_is_torch and dst_is_torch:
        logger.info("[copy_weights] Torch -> Torch")
        copy_weights_torch_to_torch(src_model, dst_model, strict)

    elif src_is_mlx and dst_is_mlx:
        logger.info("[copy_weights] MLX -> MLX")
        copy_weights_mlx_to_mlx(src_model, dst_model, strict)

    else:
        raise TypeError(
            "copy_weights: Could not detect model types. "
            "At least one of them is neither recognized as Torch nor MLX."
        )


def copy_weights_torch_to_torch(src_model, dst_model, strict: bool):
    """
    Copy all parameters from src_model to dst_model in PyTorch.
    Uses state_dict() and load_state_dict().
    """
    import torch

    src_sd = src_model.state_dict()
    if strict:
        dst_model.load_state_dict(src_sd, strict=True)
    else:
        # "Best effort" approach: filter out keys missing in dst or shape mismatches
        dst_sd = dst_model.state_dict()
        filtered_sd = {}
        for k, v in src_sd.items():
            if k in dst_sd and dst_sd[k].shape == v.shape:
                filtered_sd[k] = v
            else:
                if k not in dst_sd:
                    logger.warning(f"[Torch -> Torch] Missing key in destination: {k}")
                else:
                    logger.warning(
                        f"[Torch -> Torch] Shape mismatch for key '{k}': "
                        f"{dst_sd[k].shape} vs {v.shape}"
                    )
        dst_sd.update(filtered_sd)
        dst_model.load_state_dict(dst_sd, strict=False)

    logger.info("[copy_weights_torch_to_torch] Parameter copy complete.")


def copy_weights_mlx_to_mlx(src_model, dst_model, strict: bool):
    import logging
    import mlx.core as mx
    from mlx.utils import tree_flatten

    logger = logging.getLogger(__name__)

    src_params = dict(tree_flatten(src_model.parameters()))
    dst_params = dict(tree_flatten(dst_model.parameters()))

    for k, src_array in src_params.items():
        if k not in dst_params:
            if strict:
                raise KeyError(f"[MLX -> MLX] Destination missing key: {k}")
            else:
                logger.warning(f"[MLX -> MLX] Skipping missing key in destination: {k}")
                continue

        # shape check if strict
        if strict and (dst_params[k].shape != src_array.shape):
            raise ValueError(
                f"[MLX -> MLX] Shape mismatch for key '{k}': "
                f"dst={dst_params[k].shape} vs src={src_array.shape}"
            )

        # Rebind the dictionary entry to the src_array
        # This means dst_params[k] now directly references the same MLX array object
        # from the source.
        dst_params[k] = src_array

    logger.info("[copy_weights_mlx_to_mlx] Parameter copy complete.")
