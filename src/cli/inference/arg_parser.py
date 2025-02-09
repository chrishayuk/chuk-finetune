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
        default=None,
        help="A system prompt giving high-level context (only relevant if --chat)."
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

    # Sampler argument
    parser.add_argument(
        "--sampler",
        type=str,
        default="default",
        choices=["default", "top_p"],
        help="Sampler type: 'default' (greedy) or 'top_p' (nucleus sampling)."
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

    # stop_sequences argument
    parser.add_argument(
        "--stop_sequences",
        type=str,
        default="<|endoftext|>",
        help="Comma-separated list of stop sequences to use. "
             'Example: --stop_sequences "</answer>,User:,Assistant:"'
    )

    # number of responses
    parser.add_argument(
        "--num_responses",
        type=int,
        default=1,
        help="Number of responses to generate in single-turn mode. "
             "If >1 and sampler=top_p, we do multiple top-p samples."
    )

    # parse and return
    return parser.parse_args()
