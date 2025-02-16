#!/usr/bin/env python
"""
teacher_collection_cli.py

A CLI script that runs a single-pass teacher data collection pipeline.
It:
  - Loads the teacher model & tokenizer (via teacher_model_loader).
  - Loads an input dataset from a JSONL file (one JSON object per line).
  - Optionally applies a prompt template to each dataset prompt (with a default provided).
  - Uses an integrated reward function (wrapping combined_calculate_reward from verifiers).
  - Collects teacher outputs using collect_teacher_data_once().
  - Keeps only the single best response per item if --keep_best is set.
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
from train.teacher.teacher_data_collector import collect_teacher_data_once

# Verifiers
from verifiers.combined_reward import combined_calculate_reward

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Default prompt template
DEFAULT_PROMPT_TEMPLATE = (
    "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
    "The assistant first thinks about the reasoning process in the think tags and then provides the user "
    "with the answer in a user friendly manner within the answer tags, and finally also provides just the "
    "answer within the verifier tags, so it can be checked by an automated process. The reasoning process, "
    "answer and verifier answer are enclosed within <think> </think>, <answer> </answer>, and "
    "<verifier_answer> </verifier_answer> tags, respectively, i.e., "
    "<think>reasoning process here</think><answer>user answer here</answer><verifier_answer>verifier answer here</verifier_answer>. "
    "User: {{question}} Assistant:"
)

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
    parser.add_argument(
        "--prompt_template",
        type=str,
        default=DEFAULT_PROMPT_TEMPLATE,
        help=(
            "A string template for each item's prompt. Must contain '{{question}}' as a placeholder. "
            "If not provided, the default template includes: \n\n"
            f"{DEFAULT_PROMPT_TEMPLATE}\n\n"
            "You can override it by passing --prompt_template \"Your template here...\""
        )
    )
    parser.add_argument(
        "--keep_best",
        action="store_true",
        help="If set, keep only the single best (highest-reward) response per item in the final output."
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
        teacher_model, tokenizer, device = load_teacher_model(
            args.model, device_override=args.device
        )
    except RuntimeError as e:
        logger.error("Failed to load teacher model.")
        logger.error(f"Error details: {e}")
        sys.exit(1)

    logger.info(f"Using device: {device}")

    # Load dataset from JSONL file.
    dataset = load_dataset(args.dataset)

    # Define integrated reward function (wrapping combined_calculate_reward).
    def integrated_reward(response_text, item):
        score, feedback = combined_calculate_reward(response_text, item)
        logger.info(
            f"Reward for response '{response_text}' on item '{item}': "
            f"score={score}, feedback={feedback}"
        )
        return score, feedback

    # Collect teacher data.
    final_data = collect_teacher_data_once(
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        dataset=dataset,
        calculate_reward=integrated_reward,
        batch_size=args.batch_size,
        G=args.G,
        device=device,
        verbose=True,
        generate_single_fn=generate_single_teacher_response,
        prompt_template=args.prompt_template
    )

    if not final_data:
        logger.warning("No data collected. Check your dataset and reward function settings.")
    else:
        logger.info(f"Collected {len(final_data)} teacher data items.")

    # If --keep_best is set, reduce each item to the single best response
    if args.keep_best:
        logger.info("Reducing dataset to single best response per item...")
        final_data = keep_best_response_only(final_data)
        logger.info(f"After distillation, we have {len(final_data)} items left.")

    # Save the collected (and possibly distilled) teacher data as JSONL.
    save_dataset_jsonl(final_data, args.output)

if __name__ == "__main__":
    main()

def keep_best_response_only(collected_data):
    """
    For each item in 'collected_data', keep only the single
    best (highest-reward) response, discarding others.

    Args:
        collected_data: A list of items from your teacher pipeline.
            Each item is typically a dict with:
              {
                "item": { ... },
                "responses": [list of str],
                "teacher_logprobs": [list of float],
                "rewards": [list of float],
                "feedbacks": [list of str],
              }
    
    Returns:
        A new list of items, each containing only one (best) response.
    """
    distilled = []
    for entry in collected_data:
        rewards = entry.get("rewards", [])
        if not rewards:
            # No responses or rewards, skip entirely
            continue
        
        # Identify the highest-reward response index
        max_idx = max(range(len(rewards)), key=lambda i: rewards[i])

        # Build a new structure containing only that best response
        best_item = {
            "item": entry["item"],
            "response": entry["responses"][max_idx],
            "reward": rewards[max_idx],
            "teacher_logprob": entry["teacher_logprobs"][max_idx],
            "feedback": entry["feedbacks"][max_idx],
        }

        distilled.append(best_item)
    
    return distilled