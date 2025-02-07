#!/usr/bin/env python3
# main.py

# Training imports
from cli.train.training_monitor import monitor_training_progress
from reward_functions import combined_calculate_reward, set_eval_model
from train.grpo.grpo_trainer import train_grpo

# CLI imports
from cli.train.arg_parser import parse_arguments
from cli.train.logger_config import YELLOW, logger, color_text, BOLD, GREEN
from cli.train.model_loader import load_models

# Dataset and prompt-handling imports
from cli.train.verifiers_dataset_loader import load_prompts_and_verifiers
from cli.train.prompt_handler import prepare_prompts

def main():
    # Parse CLI arguments.
    args = parse_arguments()

    # Load the training models.
    logger.info(f"Base Model: {args.model}, Device: {args.device or 'Auto-Detect'}")
    base_model, ref_model, tokenizer, device = load_models(args.model, args.device)

    # Load the dataset (prompts + verifiers).
    logger.info("Loading dataset (prompts + verifiers)...")
    dataset = load_prompts_and_verifiers("dataset/zero/morse.jsonl")

    # Prepare and transform prompts
    prepared_dataset = prepare_prompts(dataset)

    # Define an integrated reward function
    def integrated_reward(response_text, item):
        return combined_calculate_reward(response_text, item)
    
    # Now we call train_grpo with as_generator=True so it yields progress events
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
        as_generator=True  # <--- IMPORTANT: we want the generator
    )

    # Pass the generator to your consumer function
    mean_loss_stage1 = monitor_training_progress(gen_stage1)
    logger.info(color_text(f"Stage One complete. Mean loss={mean_loss_stage1:.4f}", GREEN))
    logger.info(color_text("===== Training fully complete! =====", BOLD))

if __name__ == "__main__":
    main()
