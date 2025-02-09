# src/cli/train/arg_parser.py
import argparse

def parse_arguments():
    # setup the parser
    parser = argparse.ArgumentParser(description="Two-Stage RL training with format + remote checks")

    # model
    parser.add_argument("--model", type=str, required=True, help="Model name or local path.")

    # device
    parser.add_argument("--device", type=str, default=None, help="Device for training.")

    # optional arguments for adapter paths
    parser.add_argument(
        "--load-adapter-path",
        type=str,
        default=None,
        help="Path to adapter weights to load before training."
    )

    parser.add_argument(
        "--save-adapter-path",
        type=str,
        default=None,
        help="Path to save adapter weights after training."
    )

    # verbose argument
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Whether to enable verbose mode for detailed logging."
    )
    
    # parse
    return parser.parse_args()
