# src/train/teacher/teacher_model_loader.py
import logging

# model loading
from model.model_loader import load_model_and_tokenizer

# setup the logger
logger = logging.getLogger(__name__)

def load_teacher_model(
    model_name: str,
    device_override: str = None
):
    """
    Loads a teacher model & tokenizer for teacher-based data collection, 
    returning (teacher_model, tokenizer, final_device).

    Args:
        model_name (str): The path or name identifying the teacher model
                          (local path or huggingface hub name).
        device_override (str, optional): e.g. "cpu","cuda","mlx". 
                                         If None, we fallback to 'cpu' unless the model is MLX.

    Returns:
        (teacher_model, tokenizer, final_device): 
           - teacher_model: The loaded model object (MLX or Torch).
           - tokenizer: The loaded tokenizer object.
           - final_device: "mlx" if it's an MLX model, else device_override or "cpu".
    """
    logger.debug(f"Loading teacher model & tokenizer: {model_name}")

    # 1) Load the teacher model & tokenizer
    teacher_model, tokenizer, is_mlx = load_model_and_tokenizer(
        model_name_or_path=model_name,
        device_override=device_override,
    )

    # 2) If it's MLX, we might want to freeze it. (Teacher is typically not updated.)
    #    If Torch, set it to eval. This depends on your typical usage.
    if is_mlx:
        # put in eval mode (freeze it)
        logger.debug("Teacher model is MLX => freeze it.")
        teacher_model.freeze()
        
        # mlx
        final_device = "mlx"
    else:
        # freeze the model (put in eval mode)
        logger.debug("Teacher model is Torch => set to eval mode.")
        teacher_model.eval()
        
        # fallback device => user override or 'cpu'
        final_device = device_override or "cpu"

    # log it
    logger.debug(f"Teacher model => final_device={final_device}")

    # return the teacher model, tokenizer, final device
    return teacher_model, tokenizer, final_device
