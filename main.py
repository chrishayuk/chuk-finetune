#!/usr/bin/env python3
# main.py

# training imports
from prompt_handler import render_prompts
from reward_functions import calculate_reward, combined_calculate_reward
from dataset_loader import load_dataset
from train.unified_grpo_trainer import train_grpo

# cli imports
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
    args = parse_arguments()
    logger.info(f"Model: {args.model}, Device: {args.device or 'Auto-Detect'}")

    base_model, ref_model, tokenizer, device = load_models(args.model, args.device)

    logger.info("Loading dataset (poetry prompts with verifiers)...")
    dataset = load_dataset()
    data_rendered = render_prompts(dataset)

    # IMPORTANT: use the new calculate_reward that returns a tuple
    def stage_one_reward(response_text, item):
        return calculate_reward(response_text, item)

    logger.info(color_text("===== STAGE ONE: LOCAL FORMAT =====", BOLD))
    gen_stage1 = train_grpo(
        base_model, ref_model, tokenizer, data_rendered,
        stage_one_reward, 1e-5, 10, 2, 2, args.device, True
    )
    #mean_loss_stage1 = consume_training_generator(gen_stage1)
    #logger.info(color_text(f"Stage One complete. Mean loss={mean_loss_stage1:.4f}", GREEN))

    # def stage_two_reward(response_text, item):
    #     return combined_calculate_reward(response_text, item)
    #
    # logger.info(color_text("\n===== STAGE TWO: LOCAL+REMOTE =====", BOLD))
    # gen_stage2 = train_grpo(base_model, ref_model, tokenizer, data_rendered,
    #                         stage_two_reward, 1e-5, 2, 2, 2, args.device, True)
    # #mean_loss_stage2 = consume_training_generator(gen_stage2)
    # #logger.info(color_text(f"Stage Two complete. Mean loss={mean_loss_stage2:.4f}", GREEN))

    logger.info(color_text("===== Training fully complete! =====", BOLD))

if __name__ == "__main__":
    main()
