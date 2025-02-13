#!/usr/bin/env python
"""
run_teacher_collection_cli.py

A CLI script that runs a single-pass teacher data collection pipeline.
It:
  - Loads the teacher model & tokenizer (via teacher_model_loader).
  - Loads an input dataset from a JSONL file (one JSON object per line).
  - Uses an integrated reward function (wrapping combined_calculate_reward from verifiers).
  - Collects teacher outputs using collect_teacher_data_once().
  - Saves the resulting dataset as a JSONL file.

Any item with a missing 'prompt' or a reward of (None,"") is discarded.
"""

import argparse
import json
import logging
import os
import sys

# Teacher training imports
from train.teacher.teacher_generation import generate_single_teacher_response
from train.teacher.teacher_model_loader import load_teacher_model
from train.teacher.run_teacher_collection import collect_teacher_data_once

# Verifiers
from verifiers.combined_reward import combined_calculate_reward

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Run a single-pass teacher data collection pipeline using JSONL input/output."
    )
    parser.add_argument(
        "--model", type=str, required=True,
        help="Teacher model name or path (e.g., 'Qwen/Qwen2.5-3B')."
    )
    parser.add_argument(
        "--dataset", type=str, required=True,
        help="Path to the input dataset JSONL file (one JSON object per line)."
    )
    parser.add_argument(
        "--output", type=str, required=True,
        help="Path to save the collected teacher data as a JSONL file."
    )
    parser.add_argument(
        "--device", type=str, default="cpu",
        help="Device to use ('cpu', 'cuda', 'mlx'). Default is 'cpu'."
    )
    parser.add_argument(
        "--batch_size", type=int, default=2,
        help="Mini-batch size for data collection. Default is 2."
    )
    parser.add_argument(
        "--G", type=int, default=2,
        help="Number of responses to generate per item. Default is 2."
    )
    return parser.parse_args()

def load_dataset(dataset_path):
    """
    Loads a JSONL file where each line is a JSON object.
    Returns a list of items.
    """
    if not os.path.exists(dataset_path):
        logger.error(f"Dataset file {dataset_path} not found.")
        sys.exit(1)
    data = []
    with open(dataset_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except Exception as e:
                    logger.warning(f"Skipping invalid JSON line: {line}\nError: {e}")
    logger.info(f"Loaded {len(data)} items from {dataset_path}.")
    return data

def save_dataset_jsonl(data, output_path):
    """
    Saves the collected teacher data to a JSONL file.
    Each item is written as a separate JSON line.
    """
    with open(output_path, "w", encoding="utf-8") as f:
        for item in data:
            json_line = json.dumps(item)
            f.write(json_line + "\n")
    logger.info(f"Saved collected teacher data to {output_path}.")

def main():
    args = parse_arguments()

    # Load teacher model & tokenizer.
    logger.info(f"Loading teacher model & tokenizer: {args.model}")
    try:
        teacher_model, tokenizer, device = load_teacher_model(args.model, device_override=args.device)
    except RuntimeError as e:
        logger.error("Failed to load teacher model. This may be due to a missing torchvision nms operator.")
        logger.error("Please ensure that you have installed a version of torchvision that supports the 'nms' operator (e.g., torchvision>=0.15.1) or adjust your environment accordingly.")
        logger.error(f"Error details: {e}")
        sys.exit(1)
    logger.info(f"Using device: {device}")

    # Load dataset from JSONL file.
    dataset = load_dataset(args.dataset)

    # Define integrated reward function (wrapping combined_calculate_reward).
    def integrated_reward(response_text, item):
        # For debugging, you might log the reward here.
        score, feedback = combined_calculate_reward(response_text, item)
        logger.info(f"Reward for response '{response_text}' on item '{item}': score={score}, feedback={feedback}")
        return score, feedback

    # Run single-pass teacher data collection.
    final_data = collect_teacher_data_once(
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        dataset=dataset,
        calculate_reward=integrated_reward,
        batch_size=args.batch_size,
        G=args.G,
        device=device,
        verbose=True,
        generate_single_fn=generate_single_teacher_response
    )

    if not final_data:
        logger.warning("No data collected. Check your dataset and reward function settings.")
    else:
        logger.info(f"Collected {len(final_data)} teacher data items.")

    # Save the collected teacher data as JSONL.
    save_dataset_jsonl(final_data, args.output)

if __name__ == "__main__":
    main()
