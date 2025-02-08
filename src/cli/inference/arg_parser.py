# src/cli/inference/arg_parser.py
import argparse

def parse_arguments():
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(description="Simple CLI for MLX/Torch inference.")

    parser.add_argument(
        "--prompt", type=str, default=None,
        help="User prompt for single-turn mode (if not --chat)."
    )
    parser.add_argument(
        "--chat", action="store_true",
        help="Enable interactive chat mode."
    )
    parser.add_argument(
        "--system_prompt", type=str,
        default="You are a helpful assistant.",
        help="A system prompt giving high-level context."
    )
    parser.add_argument(
        "--model_name", type=str,
        default="Qwen/Qwen2.5-3B-Instruct",
        help="Model name or local path for inference."
    )
    parser.add_argument(
        "--max_new_tokens", type=int,
        default=512,
        help="Maximum number of tokens to generate."
    )
    parser.add_argument(
        "--device", type=str,
        default=None,
        help="Device for inference: cpu, cuda, mps, or mlx."
    )

    # ------------------------------------------------------------------
    # Sampler argument
    # ------------------------------------------------------------------
    parser.add_argument(
        "--sampler",
        type=str,
        default="default",
        choices=["default", "top_p"],
        help="Sampler type: 'default' (chat) or 'top_p' (nucleus sampling)."
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.6,
        help="Sampling temperature (>=0.0). Default=0.6"
    )
    parser.add_argument(
        "--top_p",
        type=float,
        default=0.95,
        help="Top-p (nucleus) sampling cutoff. Default=0.95"
    )

    return parser.parse_args()
