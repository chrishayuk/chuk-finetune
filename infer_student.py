import argparse
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    StoppingCriteria,
    StoppingCriteriaList
)

def parse_arguments():
    """
    Parses command-line arguments for the script.

    Returns:
        argparse.Namespace: The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Load a fine-tuned model (student or SFT) and generate text from a prompt."
    )
    parser.add_argument(
        "--model_path",
        type=str,
        required=True,
        help="Path to the fine-tuned model directory (e.g., './student_output' or './sft_output')."
    )
    parser.add_argument(
        "--prompt",
        type=str,
        default="Work out 50 - 123 and display the answer.",
        help="Input prompt to be provided to the model."
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cpu", "cuda", "mps", None],
        help="Override the device to use for inference: 'cpu', 'cuda', or 'mps'. If not set, auto-detection is used."
    )
    parser.add_argument(
        "--stop_word",
        type=str,
        default=None,
        help="A string that, if produced by the model, will cause generation to stop."
    )
    parser.add_argument(
        "--include_stop_word",
        action="store_true",
        help="If set, the stop word will be included in the final output."
    )
    return parser.parse_args()


def get_device(device_override=None):
    """
    Returns the device to use based on availability or an explicit override.
    
    Args:
        device_override (str, optional): If provided, forces the use of 'cpu', 'cuda', or 'mps'.
                                         Otherwise, auto-detection is performed.
    Returns:
        str: The name of the device to use (e.g., 'cpu', 'cuda', or 'mps').
    """
    if device_override:
        return device_override

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class StopWordCriteria(StoppingCriteria):
    """
    Custom stopping criteria to end generation when a specific stop_word is produced.
    """
    def __init__(self, stop_word, tokenizer):
        super().__init__()
        # Tokenize the stop_word without adding special tokens
        self.stop_ids = tokenizer.encode(stop_word, add_special_tokens=False)

    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        # If there aren't enough tokens yet to match stop_ids, continue generating
        if input_ids.shape[-1] < len(self.stop_ids):
            return False

        # Compare the tail of `input_ids` with `self.stop_ids`
        if list(input_ids[0, -len(self.stop_ids):].cpu().numpy()) == self.stop_ids:
            return True
        
        return False


def main():
    # Parse command-line arguments
    args = parse_arguments()

    # Determine device
    device = get_device(args.device)

    print(f"Loading the fine-tuned model and tokenizer from '{args.model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path).to(device)

    # Tokenize the prompt
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    # Build stopping criteria if `--stop_word` is provided
    stopping_criteria = None
    if args.stop_word:
        stopping_criteria = StoppingCriteriaList([StopWordCriteria(args.stop_word, tokenizer)])

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512,
            stopping_criteria=stopping_criteria
        )

    # Decode the generated tokens
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)

    # Optionally remove the stop word (and anything after it) from the final text
    if args.stop_word and not args.include_stop_word:
        index = generated_text.find(args.stop_word)
        if index != -1:
            generated_text = generated_text[:index]

    print("\n=== MODEL OUTPUT ===")
    print(generated_text)


if __name__ == "__main__":
    main()
