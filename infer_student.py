import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    # The directory where your fine-tuned model was saved by run_student_sft.py
    model_path = "./student_output"

    print(f"Loading the fine-tuned model and tokenizer from '{model_path}'...")
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForCausalLM.from_pretrained(model_path)

    # If you have a GPU:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # Example prompt (adapt to your training style)
    prompt = (
        "Work out 50 - 123 and display the answer. "
    )

    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=100,
            do_sample=True,
            temperature=0.7,
            top_p=0.9
        )

    # Decode
    generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\n=== MODEL OUTPUT ===")
    print(generated_text)

if __name__ == "__main__":
    main()
