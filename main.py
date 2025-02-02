#!/usr/bin/env python3
# main.py

# Training imports
from prompt_handler import render_prompts
from reward_functions import combined_calculate_reward, set_eval_model
from cli.train.dataset_loader import load_dataset
from train.unified_grpo_trainer import train_grpo

# CLI imports
from cli.train.arg_parser import parse_arguments
from cli.train.logger_config import YELLOW, logger, color_text, BOLD, GREEN
from cli.train.model_loader import load_models

def consume_training_generator(gen):
    final_mean_loss = None
    for result in gen:
        if "epoch_start" in result:
            logger.info(f"\n--- Starting epoch {result['epoch']} ---")
        elif "batch_end" in result:
            logger.info(color_text(
                f"Batch {result['batch']} ended with mean loss={result['batch_loss']:.4f}",
                YELLOW
            ))
        elif "epoch_end" in result:
            final_mean_loss = result["epoch_loss"]
            logger.info(color_text(
                f"=== Finished epoch {result['epoch']} -> mean_loss={final_mean_loss:.4f}",
                GREEN
            ))
        else:
            e = result["epoch"]
            i = result["index"]
            short_prompt = (result["prompt"][:80] + "...") if len(result["prompt"]) > 80 else result["prompt"]
            logger.info(color_text(f"\nEpoch {e}, Prompt {i} => ", BOLD) + short_prompt)
            responses = result["responses"]
            rewards = result["rewards"]
            f_loss = result["final_loss"]

            for r_idx, resp in enumerate(responses):
                logger.info(f"  Response {r_idx+1}/{len(responses)} => {resp}")
                logger.info(f"  Reward => {rewards[r_idx]:.2f}")
            logger.info(f"  Final Loss => {f_loss:.4f}")
    return final_mean_loss

def main():
    # Parse CLI arguments.
    args = parse_arguments()

    # Load the training models.
    logger.info(f"Model: {args.model}, Device: {args.device or 'Auto-Detect'}")
    base_model, ref_model, tokenizer, device = load_models(args.model, args.device)

    # Hardcode the evaluator model as "Qwen2.5-7B-Instruct".
    evaluator_model, _, evaluator_tokenizer, _ = load_models("Qwen/Qwen2.5-7B-Instruct", args.device)

    # Set the evaluator model/tokenizer for self reward calculation.
    set_eval_model(evaluator_model, evaluator_tokenizer)
    logger.info("Evaluator model set to Qwen2.5-7B-Instruct.")

    # Load the dataset.
    logger.info("Loading dataset (poetry prompts with verifiers)...")
    dataset = load_dataset("dataset/math_completions.jsonl")

    # Optionally render or mix prompts.
    data_rendered = render_prompts(dataset)
    print(data_rendered)

    # # Integrated reward function: for items containing a "completion" key,
    # # the self-reward mechanism (using the evaluator model) is used.
    # def integrated_reward(response_text, item):
    #     return combined_calculate_reward(response_text, item)

    # logger.info(color_text("===== STAGE ONE: INTEGRATED REWARD FUNCTION =====", BOLD))
    # gen_stage1 = train_grpo(
    #     base_model, ref_model, tokenizer, data_rendered,
    #     integrated_reward, 1e-5, 10, 2, 2, args.device, True
    # )

    # Uncomment to log detailed training generator output.
    # mean_loss_stage1 = consume_training_generator(gen_stage1)
    # logger.info(color_text(f"Stage One complete. Mean loss={mean_loss_stage1:.4f}", GREEN))

    logger.info(color_text("===== Training fully complete! =====", BOLD))

if __name__ == "__main__":
    main()
