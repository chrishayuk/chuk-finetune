import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import GRPO trainer, reward function, and prompt renderer
from src.train.grpo_trainer import train_grpo
from src.verifiers.response_verifier import calculate_reward
from src.prompt_renderer import PromptRenderer
from src.device_selection import DeviceSelector

def main():
    print("[INFO] Loading model and tokenizer...")
    model_name = "Qwen/Qwen2.5-3B"

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModelForCausalLM.from_pretrained(model_name)
    ref_model = AutoModelForCausalLM.from_pretrained(model_name)

    # Select device (CUDA, MPS, or CPU)
    device = DeviceSelector.get_preferred_device()
    print(f"[INFO] Using device: {device}")
    base_model.to(device)
    ref_model.to(device)
    ref_model.eval()

    print("[INFO] Loading dataset...")
    dataset_questions = [
        "What is the capital of France?",
        "Explain the concept of gravity.",
        "How to bake a chocolate cake?",
        "Tell me a joke about computers."
    ]

    print("[INFO] Rendering prompts using Jinja...")
    rendered_prompts_list = PromptRenderer.render_prompts(dataset_questions, 'src/templates/prompt_template.jinja2', as_list=True)

    # Print one example formatted prompt (for debugging)
    print("\n[INFO] Example Rendered Individual Prompt:\n")
    print(rendered_prompts_list[0])  # Debug: print first formatted prompt

    print("[INFO] Setting up verifier...")
    class MyVerifier:
        def check(self, answer: str) -> bool:
            return "paris" in (answer.lower() if answer else "")

    verifier = MyVerifier()

    print("[INFO] Beginning GRPO training...")
    train_grpo(
        base_model=base_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        verifier=verifier,
        dataset=rendered_prompts_list,  # Pass list of formatted prompts
        calculate_reward=calculate_reward,
        device=device,
        epochs=2,
        batch_size=2,
        G=2,
        lr=1e-5,
        verbose=True
    )

if __name__ == "__main__":
    main()
