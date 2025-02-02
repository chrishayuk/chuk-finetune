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
        default="Qwen/Qwen2.5-7B-Instruct",
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

    # parse
    return parser.parse_args()
