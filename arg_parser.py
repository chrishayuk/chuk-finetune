# arg_parser.py
import argparse

def parse_arguments():
    # setup the parser
    parser = argparse.ArgumentParser(description="Two-Stage RL training with format + remote checks")

    # model
    parser.add_argument("--model", type=str, required=True, help="Model name or local path.")

    # device
    parser.add_argument("--device", type=str, default=None, help="Device for training.")

    # parse
    return parser.parse_args()
