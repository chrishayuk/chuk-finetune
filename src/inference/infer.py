# src/inference/infer.py
import logging
from typing import List

# Torch-based approach
from inference.torch_inference import execute_chat_generation

# MLX-based approach
from inference.mlx_inference import mlx_chat_generation

def run_inference(
    model_name: str,
    system_prompt: str,
    user_messages: List[str],
    assistant_messages: List[str],
    max_new_tokens: int,
    device: str = None
) -> str:
    """
    A unified function that decides if we should use MLX or Torch.

    If device == 'mlx', calls 'mlx_chat_generation'.
    Otherwise, calls 'execute_chat_generation' (Torch).
    """
    if device == "mlx":
        # execute the mlx chat generation
        return mlx_chat_generation(
            model_name=model_name,
            system_prompt=system_prompt,
            user_messages=user_messages,
            max_new_tokens=max_new_tokens
        )
    else:
        logging.info(f"Using Torch inference (device={device}).")

        # execute the torch inference
        return execute_chat_generation(
            model_name=model_name,
            system_prompt=system_prompt,
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            max_new_tokens=max_new_tokens,
            device_override=device
        )
