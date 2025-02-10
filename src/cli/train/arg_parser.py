# src/cli/train/arg_parser.py
import argparse

def parse_arguments():
    parser = argparse.ArgumentParser(description="Two-Stage RL training with format + remote checks")

    # Add a config argument (optional)
    parser.add_argument(
        "--config",
        type=str,
        default="train_config.yaml",
        help="Path to YAML config file. CLI args override config values if specified."
    )

    # Model
    parser.add_argument("--model", type=str, default=None, help="Model name or local path.")

    # Device
    parser.add_argument("--device", type=str, default=None, help="Device for training.")

    # Adapter paths
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

    # Verbose argument
    parser.add_argument(
        "--verbose",
        action="store_true",
        default=False,
        help="Whether to enable verbose mode for detailed logging."
    )

    # parse arguments
    return parser.parse_args()