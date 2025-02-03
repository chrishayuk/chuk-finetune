# src/train/mlx/grpo_trainer.py

import ast
import logging
import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import generate

from train.trainer_base import Trainer  # <-- Your base Trainer interface
from train.mlx.grpo_loss import compute_advantages, grpo_loss
from train.mlx.grpo_utils import gather_logprobs, gather_kl_divergence

# Setup minimal logging format: just the message
logging.basicConfig(
    level=logging.INFO,
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
    """
    Generates a single response and computes its 'old' log-prob with the current model.
    """
    response_text = generate(
        model=model,
        tokenizer=tokenizer,
        prompt=prompt,
        max_tokens=200,
        verbose=False
    ).strip()

    if verbose:
        logger.info(color_text(f"Model: {response_text}", CYAN))

    tokens = tokenizer.encode(response_text)
    if not tokens:
        logger.warning("[WARN] Empty token sequence encountered, using fallback.")
        tokens = [tokenizer.eos_token_id]

    logits = model(mx.array(tokens, mx.uint32)[None])
    sum_lp = float(gather_logprobs(logits, tokens))
    return response_text, sum_lp

def compute_grpo_loss(
    model, 
    ref_model, 
    tokenizer, 
    item, 
    responses, 
    old_logprobs, 
    rewards, 
    kl_coeff=0.1,
    verbose=False
):
    """
    Computes a GRPO/PPO-style loss using KL from ref_model. 
    """
    advantages_arr = compute_advantages(rewards)
    if verbose:
        logger.info(f"Rewards: {rewards}")
        logger.info(f"Advantages: {advantages_arr}")

    current_list = []
    kl_list = []

    for resp in responses:
        tokens = tokenizer.encode(resp)
        if not tokens:
            logger.warning("[WARN] Empty token sequence; fallback to eos token.")
            tokens = [tokenizer.eos_token_id]

        out_current = model(mx.array(tokens, mx.uint32)[None])
        sum_current = gather_logprobs(out_current, tokens)

        out_ref = ref_model(mx.array(tokens, mx.uint32)[None])
        kl_val = gather_kl_divergence(out_current, out_ref, tokens)

        current_list.append(sum_current)
        kl_list.append(kl_val)

    logprobs_current_sums = mx.concat(current_list, axis=0)
    kl_sums = mx.concat(kl_list, axis=0)

    old_sums = mx.array(old_logprobs)
    advantages_m = mx.array(advantages_arr)

    loss_val = grpo_loss(
        logprobs_current=logprobs_current_sums,
        logprobs_old=old_sums,
        advantages=advantages_m,
        kl_divergences=kl_sums,
        kl_coeff=kl_coeff
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
    kl_coeff=0.1,
    verbose=False
):
    """
    Simple wrapper that calls 'compute_grpo_loss' for a single question.
    """
    return compute_grpo_loss(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        item=item,
        responses=responses,
        old_logprobs=old_logprobs,
        rewards=rewards,
        kl_coeff=kl_coeff,
        verbose=verbose
    )

class GRPOTrainer(Trainer):
    """
    GRPO Trainer for MLX that inherits from your base `Trainer`.

    :param model: Trainable model to be updated.
    :param ref_model: Reference model (typically fixed).
    :param tokenizer: Your tokenizer instance.
    :param optimizer: The optimizer to apply updates.
    :param calculate_reward: Callable returning (score, feedback) for a given response & item.
    :param G: Number of responses to generate per prompt.
    :param kl_coeff: Weighting for the KL penalty.
    :param device: Optional device string if relevant.
    :param verbose: Whether to log additional details.
    """

    def __init__(
        self,
        model,
        ref_model,
        tokenizer,
        optimizer,
        calculate_reward,
        G=4,
        kl_coeff=0.1,
        device=None,
        verbose=False
    ):
        super().__init__(model, tokenizer, optimizer, verbose=verbose)
        self.ref_model = ref_model
        self.calculate_reward = calculate_reward
        self.G = G
        self.kl_coeff = kl_coeff
        self.device = device  # if needed
        # Any additional initialisation if required

    def prepare_batch_data(self, batch_questions):
        """
        Takes a list of items from the data iterator (each likely { "prompt": ... })
        and returns a structure containing everything needed for `train_step`.
        """
        batch_data = []
        batch_rewards = []

        for i, raw_item in enumerate(batch_questions):
            item = ensure_dict(raw_item)
            question_str = item["prompt"].strip()

            short_q = (question_str[:100] + "...") if len(question_str) > 100 else question_str
            logger.info(color_text(f"\n=== Prompt {i} ===\n", BOLD) + short_q)

            responses = []
            old_logprobs = []
            rewards_list = []
            skip_this_item = False

            for g_idx in range(self.G):
                # Generate a single response & old logprob
                resp, old_lp = generate_single_response_and_oldlogprob(
                    model=self.model,
                    tokenizer=self.tokenizer,
                    prompt=question_str,
                    verbose=self.verbose
                )
                responses.append(resp)
                old_logprobs.append(old_lp)

                # Compute reward
                score, feedback_text = self.calculate_reward(resp, item)
                if score is None:
                    logger.info("[SKIP] Received None => skipping item.")
                    skip_this_item = True
                    break

                rewards_list.append(score)

                logger.info(color_text(f"Response {g_idx+1}/{self.G}:", GREEN) + f" {resp}")
                logger.info(f"Reward => {score:.2f}")
                logger.info(f"Verifier Feedback: {feedback_text}")

            if skip_this_item:
                continue

            avg_reward = float(np.mean(rewards_list))
            batch_rewards.append(avg_reward)

            batch_data.append({
                "item": item,
                "responses": responses,
                "old_logprobs": old_logprobs,
                "rewards": rewards_list,
            })

        return batch_data

    def train_step(self, batch_data):
        """
        Given the batch_data from `prepare_batch_data`, run the GRPO update:
          1) Summation of losses for each item in the batch
          2) Backprop and optimize
          3) Return (loss, reward).
        """
        def batch_closure(m):
            total_loss = 0.0
            valid_count = 0

            for data in batch_data:
                loss_val = single_question_loss(
                    model=m,
                    ref_model=self.ref_model,
                    tokenizer=self.tokenizer,
                    item=data["item"],
                    responses=data["responses"],
                    old_logprobs=data["old_logprobs"],
                    rewards=data["rewards"],
                    kl_coeff=self.kl_coeff,
                    verbose=self.verbose
                )
                total_loss += loss_val
                valid_count += 1

            if valid_count > 0:
                total_loss /= valid_count
            return total_loss

        if not batch_data:
            return 0.0, 0.0  # Nothing to train on

        loss_value_and_grad = nn.value_and_grad(self.model, batch_closure)
        batch_loss, grads_dict = loss_value_and_grad(self.model)

        mx.eval(grads_dict)
        self.optimizer.update(self.model, grads_dict)

        # Compute final batch reward for logging
        all_rewards = []
        for d in batch_data:
            all_rewards.extend(d["rewards"])
        final_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

        logger.info(color_text(
            f"\n[GRPO] Single Batch Update => Loss: {batch_loss:.4f}, Mean Reward: {final_reward:.4f}\n",
            YELLOW
        ))

        return float(batch_loss), final_reward

    def train_epoch(self, data_iterator, batch_size=4):
        """
        Optional convenience method if you prefer an "epoch-based" approach
        rather than a fully generic loop. Repeatedly fetches batches
        from data_iterator, calls prepare_batch_data and train_step.
        """
        all_batch_losses = []
        all_batch_rewards = []

        for batch_questions in data_iterator(batch_size):
            # Prepare data
            batch_data = self.prepare_batch_data(batch_questions)
            # Train
            batch_loss, batch_reward = self.train_step(batch_data)
            all_batch_losses.append(batch_loss)
            all_batch_rewards.append(batch_reward)

        overall_loss = float(np.mean(all_batch_losses)) if all_batch_losses else 0.0
        overall_reward = float(np.mean(all_batch_rewards)) if all_batch_rewards else 0.0

        logger.info(color_text(
            f"\n[GRPO] Epoch => Mean Loss: {overall_loss:.4f}, Mean Reward: {overall_reward:.4f}",
            GREEN
        ))
        return overall_loss, overall_reward
