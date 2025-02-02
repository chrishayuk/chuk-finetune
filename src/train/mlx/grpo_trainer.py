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

# ----- Simple ANSI Colours (optional) -----
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

def generate_single_response_and_oldlogprob(
    model,
    tokenizer,
    prompt: str,
    verbose=False
):
    response_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=200,
        verbose=False
    ).strip()

    # optional debug => changed label to "Model:"
    if verbose:
        print(color_text(f"Model: {response_text}", CYAN))

    # Handle empty tokens
    tokens = tokenizer.encode(response_text)
    if not tokens:
        logger.warning("[WARN] Empty token sequence encountered, using fallback.")
        tokens = [tokenizer.eos_token_id]

    logits = model(mx.array(tokens, mx.uint32)[None])
    sum_lp = float(gather_logprobs(logits, tokens))  # old logprob for PPO

    return response_text, sum_lp

def compute_grpo_loss(
    model,
    ref_model,
    tokenizer,
    item,
    responses,
    old_logprobs,
    rewards,
    verbose=False
):
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
    old_sums_m   = mx.array(old_logprobs)
    advantages_m = mx.array(advantages_arr)

    loss_val = grpo_loss(
        logprobs_current=logprobs_current_sums,
        logprobs_old=old_sums_m,
        advantages=advantages_m,
        kl_divergences=kl_sums
    )
    return loss_val

def single_question_loss(
    model,
    ref_model,
    tokenizer,
    item,
    responses,
    old_logprobs,
    rewards,
    verbose=False
):
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

def train_step(
    base_model,
    ref_model,
    tokenizer,
    batch_questions,
    G,
    optimizer,
    calculate_reward=None,
    device=None,
    verbose=False
):
    losses = []

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
        # Convert to dict
        item = ensure_dict(raw_item)
        question_str = item["prompt"].strip()

        # Truncate question
        short_q = (question_str[:100] + "...") if len(question_str) > 100 else question_str
        logger.info(color_text(f"\n=== Prompt {i} ===\n", BOLD) + short_q)

        responses = []
        old_logprobs = []
        rewards_list = []

        last_feedback = ""

        for g_idx in range(G):
            full_prompt = question_str
            if last_feedback:
                full_prompt += f"\n\n[Verifier Feedback]: {last_feedback}"

            resp, old_lp = generate_single_response_and_oldlogprob(
                model=base_model,
                tokenizer=tokenizer,
                prompt=full_prompt,
                verbose=verbose
            )
            responses.append(resp)
            old_logprobs.append(old_lp)

            score, feedback_text = calculate_reward(resp, item)
            rewards_list.append(score)

            # If response is trivial ("provided." or empty), skip printing
            skip_resp = (resp.lower() == "provided." or not resp.strip())
            if not skip_resp:
                logger.info(color_text(f"Response {g_idx+1}/{G}:", GREEN) + f" {resp}")

            logger.info(f"Reward => {score:.2f}")
            last_feedback = feedback_text

        # Compute GRPO loss & update
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
        logger.info(color_text(f"Final Loss => {final_loss:.4f}", YELLOW))
        losses.append(final_loss)

    return np.mean(losses)

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
    all_epoch_losses = []

    for epoch in range(epochs):
        if verbose:
            logger.info(color_text(f"\n[MLX] Starting epoch {epoch+1}/{epochs}...", CYAN))

        epoch_losses = []
        for batch_questions in data_iterator():
            # No "first batch" prints
            loss = train_step(
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
            epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses)
        all_epoch_losses.append(avg_loss)
        if verbose:
            logger.info(color_text(f"[MLX] Epoch {epoch+1} -> Mean loss: {avg_loss:.4f}", GREEN))

    return np.mean(all_epoch_losses)
