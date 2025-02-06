# src/train/model_loader.py
from cli.train.logger_config import logger, color_text, BOLD
from model.model_loader import load_model_and_tokenizer

def load_models(model_name, device_override):
    # load the base model and tokenizer
    logger.info(color_text(f"Loading base model & tokenizer: {model_name}", BOLD))

    # load the model and tokenizer
    base_model, tokenizer, device = load_model_and_tokenizer(
        model_name_or_path=model_name,
        device_override=device_override
    )

    # load the reference model and tokenizer
    logger.info("Loading reference model (KL/PPO) ...")
    ref_model, _, _ = load_model_and_tokenizer(
        model_name_or_path=model_name,
        device_override=device_override
    )

    # check if we're not mlx
    if device_override != "mlx" and device is not None:
        # set the device for reference model
        ref_model.to(device)
        ref_model.eval()
    else:
        # freeze the model (equivalent to eval in mlx)
        ref_model.freeze()

    # return the base model, reference model, tokenizer and device
    return base_model, ref_model, tokenizer, device
