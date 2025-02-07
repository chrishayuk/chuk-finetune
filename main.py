#!/usr/bin/env python3
# main.py

# Standard library
import os

# Training imports
from reward_functions import combined_calculate_reward, set_eval_model
from train.grpo.grpo_trainer import train_grpo

# CLI imports
from cli.train.arg_parser import parse_arguments
from cli.train.logger_config import YELLOW, logger, color_text, BOLD, GREEN
from cli.train.model_loader import load_models
from cli.train.training_monitor import monitor_training_progress
from cli.train.prompt_handler import prepare_prompts
from cli.train.verifiers_dataset_loader import load_prompts_and_verifiers

# adapters
from model.adapters import save_adapters, load_adapters

def main():
    # Parse CLI arguments.
    args = parse_arguments()

    # Load the training models.
    logger.info(f"Base Model: {args.model}, Device: {args.device or 'Auto-Detect'}")
    base_model, ref_model, tokenizer, device = load_models(args.model, args.device)

    # --- (1) Load adapters if user specified a path ---
    if args.load_adapter_path is not None:
        if not os.path.isfile(args.load_adapter_path):
            raise FileNotFoundError(f"Could not find adapter file: {args.load_adapter_path}")
        logger.info(f"Loading adapters from {args.load_adapter_path}...")
        load_adapters(base_model, args.load_adapter_path)

    # Load the dataset (prompts + verifiers).
    logger.info("Loading dataset (prompts + verifiers)...")
    dataset = load_prompts_and_verifiers("dataset/zero/verifier_samples_very_easy.jsonl")

    # Prepare and transform prompts
    prepared_dataset = prepare_prompts(dataset)

    # Define an integrated reward function
    def integrated_reward(response_text, item):
        return combined_calculate_reward(response_text, item)
    
    # --- (2) Train the model via GRPO ---
    gen_stage1 = train_grpo(
        base_model=base_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=prepared_dataset,
        calculate_reward=integrated_reward,
        lr=1e-6,         # Learning rate
        epochs=10,       # Number of epochs
        batch_size=2,    # Batch size
        G=2,             # Generate 2 responses per prompt
        device=args.device,
        verbose=True,
        as_generator=False  # Not returning a generator in this snippet
    )

    # (3) Pass the returned object to your consumer function for logging
    mean_loss_stage1 = monitor_training_progress(gen_stage1)
    logger.info(color_text(f"Stage One complete. Mean loss={mean_loss_stage1:.4f}", GREEN))
    logger.info(color_text("===== Training fully complete! =====", BOLD))

    # --- (4) Save adapters if user specified a path ---
    if args.save_adapter_path is not None:
        logger.info(f"Saving adapters to {args.save_adapter_path}...")
        save_adapters(base_model, args.save_adapter_path)

if __name__ == "__main__":
    main()
