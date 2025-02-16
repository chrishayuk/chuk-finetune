import argparse
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
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
    args = parser.parse_args()

    print(f"Loading the fine-tuned model and tokenizer from '{args.model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(args.model_path)

    # Move model to GPU if available
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Tokenize the prompt
    inputs = tokenizer(args.prompt, return_tensors="pt").to(device)

    # Generate output
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=512
        )

    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n=== MODEL OUTPUT ===")
    print(generated_text)

if __name__ == "__main__":
    main()
