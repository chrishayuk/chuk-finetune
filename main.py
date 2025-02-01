# main.py

import argparse

# Our imports
from prompt_renderer import PromptRenderer
from model_utils import load_model_and_tokenizer

# >>>> IMPORTANT: Import from your unified GRPO trainer <<<<
from src.train.unified_grpo_trainer import train_grpo
from verifiers.response_verifier import calculate_reward

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Load models via Torch or MLX for training."
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        help="Model name or local path to load."
    )

    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use for training. Use 'mlx' for Apple MLX, or 'cpu', 'cuda', 'mps' (Torch)."
    )
    return parser.parse_args()

def load_models(model_name: str, device_override: str):
    """
    Load base and reference models + tokenizer via either MLX or Torch
    depending on device_override. 
    If device_override == 'mlx', we do MLX logic.
    Otherwise, Torch logic.
    """
    print("[INFO] Loading base model & tokenizer...")
    base_model, tokenizer, device = load_model_and_tokenizer(
        model_name_or_path=model_name,
        device_override=device_override
    )

    print("[INFO] Loading reference model...")
    ref_model, _, _ = load_model_and_tokenizer(
        model_name_or_path=model_name,
        device_override=device_override
    )
    # For Torch, set reference model device & eval
    if device_override != "mlx" and device is not None:
        ref_model.to(device)
        ref_model.eval()

    return base_model, ref_model, tokenizer, device

def load_dataset():
    """Load or define dataset questions."""
    print("[INFO] Loading dataset...")
    return [
        "What is the capital of France?",
        "Explain the concept of gravity.",
        "How to bake a chocolate cake?",
        "Tell me a joke about computers."
    ]

def render_prompts(dataset):
    """Render prompts using Jinja."""
    print("[INFO] Rendering prompts...")
    rendered_prompts = PromptRenderer.render_prompts(
        dataset,
        'src/templates/prompt_template.jinja2',
        as_list=True
    )
    print("\n[INFO] Example Rendered Prompt:\n")
    print(rendered_prompts[0])
    return rendered_prompts

def main():
    """Main function to load models, dataset, and prompts, then do GRPO training."""
    args = parse_arguments()
    print(f"[INFO] Loading model: {args.model}")
    print(f"[INFO] Device: {args.device or 'Auto-Detect (Torch)'}")

    # 1) Load base & reference models + tokenizer
    base_model, ref_model, tokenizer, device = load_models(args.model, args.device)

    # 2) Load or create dataset
    dataset = load_dataset()

    # 3) Render prompts
    rendered_prompts = render_prompts(dataset)

    # 4) GRPO training
    class MyVerifier:
        def check(self, answer: str) -> bool:
            return "paris" in (answer.lower() if answer else "")

    verifier = MyVerifier()

    print("[INFO] Beginning GRPO training...")
    mean_loss = train_grpo(
        base_model=base_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        verifier=verifier,
        dataset=rendered_prompts,
        calculate_reward=calculate_reward,
        lr=1e-5,
        epochs=2,
        batch_size=2,
        G=2,
        device=args.device,
        verbose=True
    )
    print(f"[INFO] Training complete. Mean loss: {mean_loss:.4f}")

if __name__ == "__main__":
    main()
