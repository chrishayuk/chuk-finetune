# src/model_utils.py
from transformers import AutoModelForCausalLM, AutoTokenizer

# imports
from device_selection import DeviceSelector

def load_model_and_tokeniser(model_name: str, device_override: str):
    """
    Load a Qwen model and its tokeniser onto the appropriate device.
    """
    # load the device
    device = DeviceSelector.get_preferred_device(device_override)

    # load the model
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype="auto").to(device)

    # load the tokenizer
    tokeniser = AutoTokenizer.from_pretrained(model_name)

    # return the model, tokenizer and device
    return model, tokeniser, device
