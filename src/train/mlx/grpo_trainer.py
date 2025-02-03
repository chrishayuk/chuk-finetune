import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import generate

from train.mlx.grpo_loss import compute_advantages, grpo_loss
from train.mlx.grpo_utils import gather_logprobs, gather_kl_divergence

import ast
import logging

# Setup minimal logging format: just the message
logging.basicConfig(
    level=logging.INFO,  # Adjust to DEBUG if you need more details
    format="%(message)s",
)
logger = logging.getLogger(__name__)

RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"

def color_text(text, color):
    return f"{color}{text}{RESET}"

def ensure_dict(item):
    if isinstance(item, dict):
        return item
    if isinstance(item, str) and item.strip().startswith("{"):
        try:
            possible_dict = ast.literal_eval(item)
            if isinstance(possible_dict, dict):
                return possible_dict
        except:
            pass
    raise ValueError(f"[ERROR] Unexpected non-dict item: {item}")

def generate_single_response_and_oldlogprob(model, tokenizer, prompt: str, verbose=False):
    response_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=200,
        verbose=False
    ).strip()

    if verbose:
        print(color_text(f"Model: {response_text}", CYAN))

    tokens = tokenizer.encode(response_text)
    if not tokens:
        logger.warning("[WARN] Empty token sequence encountered, using fallback.")
        tokens = [tokenizer.eos_token_id]

    logits = model(mx.array(tokens, mx.uint32)[None])
    sum_lp = float(gather_logprobs(logits, tokens))  # old logprob for PPO

    return response_text, sum_lp

def compute_grpo_loss(model, ref_model, tokenizer, item, responses, old_logprobs, rewards, verbose=False):
    advantages_arr = compute_advantages(rewards)
    if verbose:
        logger.info(f"Rewards: {rewards}")
        logger.info(f"Advantages: {advantages_arr}")

    current_list = []
    kl_list = []
    for resp in responses:
        tokens = tokenizer.encode(resp)
        if not tokens:
            logger.warning("[WARN] Empty token sequence in compute_grpo_loss, fallback.")
            tokens = [tokenizer.eos_token_id]

        out_current = model(mx.array(tokens, mx.uint32)[None])
        sum_current = gather_logprobs(out_current, tokens)

        out_ref = ref_model(mx.array(tokens, mx.uint32)[None])
        kl_val = gather_kl_divergence(out_current, out_ref, tokens)

        current_list.append(sum_current)
        kl_list.append(kl_val)

    logprobs_current_sums = mx.concat(current_list, axis=0)
    kl_sums = mx.concat(kl_list, axis=0)
    old_sums_m = mx.array(old_logprobs)
    advantages_m = mx.array(advantages_arr)

    loss_val = grpo_loss(
        logprobs_current=logprobs_current_sums,
        logprobs_old=old_sums_m,
        advantages=advantages_m,
        kl_divergences=kl_sums
    )
    return loss_val

def single_question_loss(model, ref_model, tokenizer, item, responses, old_logprobs, rewards, verbose=False):
    return compute_grpo_loss(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        item=item,
        responses=responses,
        old_logprobs=old_logprobs,
        rewards=rewards,
        verbose=verbose
    )

def train_step(base_model, ref_model, tokenizer, batch_questions, G, optimizer, calculate_reward=None, device=None, verbose=False):
    """
    Processes a single batch of questions with GRPO training.
    If *any* generated response yields a None reward, we skip that item entirely.
    """
    losses = []
    batch_rewards = []
    batch_idx = 0

    def closure(model, item, responses, old_logprobs, rewards):
        return single_question_loss(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            item=item,
            responses=responses,
            old_logprobs=old_logprobs,
            rewards=rewards,
            verbose=verbose
        )

    loss_value_and_grad = nn.value_and_grad(base_model, closure)

    for i, raw_item in enumerate(batch_questions):
        item = ensure_dict(raw_item)
        question_str = item["prompt"].strip()

        short_q = (question_str[:100] + "...") if len(question_str) > 100 else question_str
        logger.info(color_text(f"\n=== Prompt {i} ===\n", BOLD) + short_q)

        responses = []
        old_logprobs = []
        rewards_list = []

        skip_this_item = False

        for g_idx in range(G):
            full_prompt = question_str
            resp, old_lp = generate_single_response_and_oldlogprob(
                model=base_model,
                tokenizer=tokenizer,
                prompt=full_prompt,
                verbose=verbose
            )
            responses.append(resp)
            old_logprobs.append(old_lp)

            score, feedback_text = calculate_reward(resp, item)
            
            # If reward is None => skip training for this item
            if score is None:
                logger.info("[SKIP] Verifier unavailable or error => skipping item.")
                skip_this_item = True
                break

            rewards_list.append(score)

            logger.info(color_text(f"Response {g_idx+1}/{G}:", GREEN) + f" {resp}")
            logger.info(f"Reward => {score:.2f}")
            logger.info(f"Verifier Feedback: {feedback_text}")

        # Skip if any response had None
        if skip_this_item:
            continue

        avg_reward = np.mean(rewards_list)
        batch_rewards.append(avg_reward)

        loss_val, grads_dict = loss_value_and_grad(
            base_model,
            item,
            responses,
            old_logprobs,
            rewards_list
        )
        mx.eval(grads_dict)
        optimizer.update(base_model, grads_dict)

        final_loss = float(loss_val)
        losses.append(final_loss)
        batch_idx += 1
        logger.info(color_text(f"Batch {batch_idx}: Loss => {final_loss:.4f}, Mean Reward => {avg_reward:.4f}", YELLOW))

    # If the entire batch is skipped, let's avoid NaN by returning 0.0
    mean_loss = np.mean(losses) if losses else 0.0
    mean_reward = np.mean(batch_rewards) if batch_rewards else 0.0
    return mean_loss, mean_reward

def train_grpo(
    base_model,
    ref_model,
    tokenizer,
    data_iterator,
    calculate_reward,
    optimizer,
    epochs: int = 1,
    batch_size: int = 4,
    G: int = 4,
    device=None,
    verbose=False
):
    """
    Outer training function that runs for the specified number of epochs,
    feeding batches from 'data_iterator' into 'train_step'.
    """
    all_batch_losses = []
    all_batch_rewards = []

    for epoch in range(epochs):
        if verbose:
            logger.info(color_text(f"\n[MLX] Starting epoch {epoch+1}/{epochs}...", CYAN))
        for batch_questions in data_iterator():
            batch_loss, batch_reward = train_step(
                base_model=base_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                batch_questions=batch_questions,
                G=G,
                optimizer=optimizer,
                calculate_reward=calculate_reward,
                device=device,
                verbose=verbose
            )
            all_batch_losses.append(batch_loss)
            all_batch_rewards.append(batch_reward)

    overall_loss = np.mean(all_batch_losses) if all_batch_losses else 0.0
    overall_reward = np.mean(all_batch_rewards) if all_batch_rewards else 0.0
    logger.info(color_text(f"Overall Mean Loss: {overall_loss:.4f}", GREEN))
    logger.info(color_text(f"Overall Mean Reward: {overall_reward:.4f}", GREEN))
    return overall_loss, overall_reward
