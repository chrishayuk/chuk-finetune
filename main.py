#!/usr/bin/env python3
# main.py
import argparse
import logging
import requests
import ast
import sys

from prompt_renderer import PromptRenderer
from model_utils import load_model_and_tokenizer
from src.dataset_loader import load_dataset
from src.train.unified_grpo_trainer import train_grpo  # This trainer yields generator results
from src.verifiers.response_verifier import check_format

# ANSI Colours (optional)
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
RED = "\033[91m"
CYAN = "\033[96m"

def color_text(text, color):
    return f"{color}{text}{RESET}"

###############################################################################
# Logging
###############################################################################
logging.basicConfig(
    level=logging.DEBUG,  # or logging.INFO for less verbosity
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

###############################################################################
# Two Reward Functions
###############################################################################
def local_format_calculate_reward(response_text: str, item: dict) -> float:
    """
    1) 1.0 if <think>, <answer>, <verifier_answer> tags are present,
       else 0.0.  (No remote checks)
    """
    format_score = check_format(response_text)
    return format_score

def remote_calculate_reward(response_text: str, item: dict):
    """
    Calls a remote verifier endpoint, returning (score, feedback).
    """
    url = item.get("verifier_url", "http://0.0.0.0:8000") + "/verify"
    payload = {
        "text": response_text,
        "verifier": item["verifier"],
        "feedback": True
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        score = data.get("score", 0.0)
        feedback = " ".join(data.get("feedback", []))
        logger.debug(color_text(f"[Remote] score={score}, feedback={feedback}", CYAN))
        return score, feedback
    except Exception as e:
        logger.warning(f"Remote verifier error: {e}")
        return 0.0, "Error: Could not reach verifier."

def combined_calculate_reward(response_text: str, item: dict):
    """
    Adds local format reward + remote reward => total float, plus feedback string.
    """
    local_score = local_format_calculate_reward(response_text, item)
    remote_score, remote_fb = remote_calculate_reward(response_text, item)
    total = local_score + remote_score
    logger.debug(color_text(
        f"[combined_calculate_reward] local={local_score:.2f}, remote={remote_score:.2f}, total={total:.2f}",
        YELLOW
    ))
    return total, remote_fb

###############################################################################
# Argument Parsing
###############################################################################
def parse_arguments():
    parser = argparse.ArgumentParser(description="Two-Stage RL training with format + remote checks")
    parser.add_argument("--model", type=str, required=True, help="Model name or local path.")
    parser.add_argument("--device", type=str, default=None, help="Device for training.")
    return parser.parse_args()

###############################################################################
# Load Models
###############################################################################
def load_models(model_name, device_override):
    logger.info(color_text(f"Loading base model & tokenizer: {model_name}", BOLD))
    base_model, tokenizer, device = load_model_and_tokenizer(
        model_name_or_path=model_name,
        device_override=device_override
    )
    logger.info("Loading reference model (KL/PPO) ...")
    ref_model, _, _ = load_model_and_tokenizer(
        model_name_or_path=model_name,
        device_override=device_override
    )
    if device_override != "mlx" and device is not None:
        ref_model.to(device)
        ref_model.eval()
    return base_model, ref_model, tokenizer, device

###############################################################################
# Render Prompts
###############################################################################
def render_prompts(dataset):
    # Convert each item’s "prompt" with Jinja
    rendered = []
    for item in dataset:
        prompt_text = PromptRenderer.render_prompts(
            [item["prompt"]],
            "src/templates/prompt_template.jinja2",
            as_list=True
        )[0]
        rendered.append({
            "prompt": prompt_text,
            "verifier": item.get("verifier", "haiku"),
            "verifier_url": item.get("verifier_url", "http://0.0.0.0:8000")
        })
    return rendered

###############################################################################
# Helper: Consume Training Generator
###############################################################################
def consume_training_generator(gen):
    """
    If train_grpo(...) yields dictionaries:
      - epoch_start => { "epoch_start": True, "epoch": e }
      - batch_end   => { "batch_end": True, "epoch": e, "batch": b, "batch_loss": float }
      - epoch_end   => { "epoch_end": True, "epoch": e, "epoch_loss": float }
      - item result => { "epoch": e, "index": i, "prompt":..., "responses":..., "rewards":..., "final_loss":... }
    we handle them in real time here.

    Returns the final epoch’s mean_loss for convenience.
    """
    final_mean_loss = None
    for result in gen:
        if "epoch_start" in result:
            logger.info(f"\n--- Starting epoch {result['epoch']} ---")

        elif "batch_end" in result:
            # Per-batch stats
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
            # Item-level result
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

###############################################################################
# Main
###############################################################################
def main():
    args = parse_arguments()
    logger.info(f"Model: {args.model}, Device: {args.device or 'Auto-Detect'}")

    # 1) Load
    base_model, ref_model, tokenizer, device = load_models(args.model, args.device)
    # 2) Load dataset
    logger.info("Loading dataset (poetry prompts with verifiers)...")
    dataset = load_dataset()
    # 3) Render
    data_rendered = render_prompts(dataset)

    # == Stage 1: Local Format Only
    def stage_one_reward(response_text, item):
        s = local_format_calculate_reward(response_text, item)
        return s, "No remote feedback"

    logger.info(color_text("===== STAGE ONE: LOCAL FORMAT =====", BOLD))
    # We call train_grpo(...), which returns a generator (yielding item, batch, epoch info)
    gen_stage1 = train_grpo(
        base_model=base_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=data_rendered,
        calculate_reward=stage_one_reward,
        lr=1e-5,
        epochs=2,
        batch_size=2,
        G=2,
        device=args.device,
        verbose=True
    )

    mean_loss_stage1 = consume_training_generator(gen_stage1)
    logger.info(color_text(f"Stage One complete. Mean loss={mean_loss_stage1:.4f}", GREEN))

    # == Stage 2: Combined Local+Remote
    def stage_two_reward(response_text, item):
        total, fb = combined_calculate_reward(response_text, item)
        return total, fb

    logger.info(color_text("\n===== STAGE TWO: LOCAL+REMOTE =====", BOLD))
    gen_stage2 = train_grpo(
        base_model=base_model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        dataset=data_rendered,
        calculate_reward=stage_two_reward,
        lr=1e-5,
        epochs=2,
        batch_size=2,
        G=2,
        device=args.device,
        verbose=True
    )

    mean_loss_stage2 = consume_training_generator(gen_stage2)
    logger.info(color_text(f"Stage Two complete. Mean loss={mean_loss_stage2:.4f}", GREEN))

    logger.info(color_text("===== Training fully complete! =====", BOLD))


if __name__ == "__main__":
    main()
