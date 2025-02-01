# src/inference/infer.py

import logging
from typing import List

from inference.torch.torch_inference import execute_chat_generation
from inference.mlx.mlx_inference import mlx_chat_generation

def load_inference_model(model_name: str, device: str = None):
    """
    Loads a model + tokenizer once, returning (model, tokenizer, is_mlx).

    If device == 'mlx', load Apple MLX; otherwise, load Torch.
    """
    if device == "mlx":
        logging.info("Loading MLX model once...")
        from mlx_lm import load as mlx_load
        model, tokenizer = mlx_load(model_name)
        is_mlx = True
    else:
        logging.info(f"Loading Torch model (device={device or 'auto'}) once...")
        from model_utils import load_model_and_tokenizer
        model, tokenizer, _dev = load_model_and_tokenizer(
            model_name_or_path=model_name,
            device_override=device
        )
        is_mlx = False

    return model, tokenizer, is_mlx

def run_inference(
    model,
    tokenizer,
    is_mlx: bool,
    system_prompt: str,
    user_messages: List[str],
    assistant_messages: List[str],
    max_new_tokens: int
):
    """
    Execute inference using a *preloaded* model/tokenizer.

    If is_mlx=True, calls MLX's 'mlx_chat_generation_preloaded'.
    Otherwise, calls Torch's 'execute_chat_generation_preloaded'.

    This approach avoids re-loading or re-downloading the model on every call.

    :param model: A preloaded MLX or Torch model object.
    :param tokenizer: A preloaded tokenizer object.
    :param is_mlx: Boolean indicating if the loaded model is MLX (True) or Torch (False).
    :param system_prompt: An optional system instruction for context.
    :param user_messages: List of user messages.
    :param assistant_messages: List of assistant messages.
    :param max_new_tokens: Maximum tokens to generate.
    :return: The generated assistant response (string).
    """
    if is_mlx:
        logging.info("Using preloaded MLX model for inference.")
        return mlx_chat_generation(
            loaded_model=model,
            loaded_tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            max_new_tokens=max_new_tokens
        )
    else:
        logging.info("Using preloaded Torch model for inference.")
        return execute_chat_generation(
            loaded_model=model,
            loaded_tokenizer=tokenizer,
            system_prompt=system_prompt,
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            max_new_tokens=max_new_tokens
        )
