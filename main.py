# main.py
import argparse

# Our imports
from src.prompt_renderer import PromptRenderer
from src.model_utils import load_model_and_tokenizer

def parse_arguments():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Load models via Torch or MLX for training.")
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
        help=(
            "Device to use for training. "
            "Use 'mlx' for Apple MLX, or 'cpu', 'cuda', 'mps' (Torch). "
            "If None, Torch auto-selects (GPU if available, else CPU)."
        )
    )
    return parser.parse_args()

def load_models(model_name: str, device_override: str):
    """
    Load base and reference models along with tokenizer, using either MLX or Torch.

    If device_override == 'mlx', we do MLX logic.
    Otherwise, we do Torch logic (CPU, CUDA, MPS, etc.).
    """
    print("[INFO] Loading base model & tokenizer...")
    base_model, tokenizer, device = load_model_and_tokenizer(
        model_name_or_path=model_name,
        device_override=device_override
    )

    print("[INFO] Loading reference model...")
    ref_model, _, ref_dev = load_model_and_tokenizer(
        model_name_or_path=model_name,
        device_override=device_override
    )
    # For Torch, we set the reference model to the same device (and eval mode)
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
    """Main function to load models, dataset, and prompts, and optionally do GRPO training."""
    args = parse_arguments()
    print(f"[INFO] Loading model: {args.model}")
    print(f"[INFO] Device: {args.device or 'Auto-Detect (Torch)'}")

    # 1) Load base and reference models
    base_model, ref_model, tokenizer, device = load_models(args.model, args.device)

    # 2) Load or create dataset
    dataset = load_dataset()

    # 3) Render prompts
    rendered_prompts = render_prompts(dataset)

    # 4) (Optional) GRPO training
    # -------------------------------------------------
    from src.train.grpo_trainer import train_grpo
    from src.verifiers.response_verifier import calculate_reward
    
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
        dataset=rendered_prompts,
        calculate_reward=calculate_reward,
        device=device,
        epochs=2,
        batch_size=2,
        G=2,
        lr=1e-5,
        verbose=True
    )
    # -------------------------------------------------

if __name__ == "__main__":
    main()
