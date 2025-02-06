# src/train/grpo/mlx/grpo_trainer.py
import ast
import logging
import numpy as np
import mlx.core as mx
import mlx.nn as nn

# imports
from train.grpo.mlx.grpo_generation import generate_single_response_and_oldlogprob
from train.trainer_base import Trainer
from train.grpo.mlx.grpo_loss import single_question_loss

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

class GRPOTrainer(Trainer):
    """
    GRPO Trainer for MLX
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
        # call parent constructor
        super().__init__(model, tokenizer, optimizer, verbose=verbose)

        # set properties
        self.ref_model = ref_model
        self.calculate_reward = calculate_reward
        self.G = G
        self.kl_coeff = kl_coeff
        self.device = device

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
