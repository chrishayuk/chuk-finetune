import ast
import logging
import numpy as np
import mlx.core as mx
import mlx.nn as nn

# Local imports
from train.grpo.mlx.grpo_generation import generate_single_response_and_oldlogprob
from train.grpo.mlx.grpo_loss import single_question_loss
from train.trainer_base import Trainer

# Setup minimal logging format: just the message
logging.basicConfig(level=logging.INFO, format="%(message)s")
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
    """
    Attempt to ensure the 'item' is a dict. If it's already a dict, return it.
    If it's a string that looks like JSON/dict, parse it.
    Otherwise, raise an error.
    """
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
    GRPO Trainer for MLX framework.

    Expects:
      - self.model / self.ref_model: MLX-compatible models
      - self.optimizer: MLX-compatible optimizer
      - self.calculate_reward: function(resp_text, item_dict) => (score, feedback_text)
      - G: number of response samples per item
      - kl_coeff: KL penalty weight
      - device: for MLX, might be True/None if no explicit GPU device logic needed
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
        # Call the parent Trainer constructor
        super().__init__(model, tokenizer, optimizer, device=device, verbose=verbose)

        self.ref_model = ref_model
        self.calculate_reward = calculate_reward
        self.G = G
        self.kl_coeff = kl_coeff

    def prepare_batch_data(self, batch_questions):
        """
        Takes a list of items (each typically { "prompt": ... }) and returns
        a structure containing everything needed for train_step.

        For each item:
          1) Generate G responses from the model
          2) Compute reward
          3) Collect old logprob, etc.

        We'll skip items where reward is None (signifying invalid or no feedback).
        """
        batch_data = []

        for i, raw_item in enumerate(batch_questions):
            # Attempt to parse the item into a dict
            item = ensure_dict(raw_item)
            question_str = item["prompt"].strip()

            short_q = (question_str[:100] + "...") if len(question_str) > 100 else question_str
            logger.info(color_text(f"\n=== Prompt {i} ===\n", BOLD) + short_q)

            responses = []
            old_logprobs = []
            rewards_list = []
            skip_this_item = False

            for g_idx in range(self.G):
                # Generate one response & old logprob
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
                # if any response was invalid => skip entire item
                continue

            # Construct a single data record
            batch_data.append({
                "item": item,
                "responses": responses,
                "old_logprobs": old_logprobs,
                "rewards": rewards_list,
            })

        return batch_data

    def train_step(self, batch_data):
        """
        Perform a GRPO update:
          - For each item in batch_data, compute a single_question_loss
          - Accumulate and average loss
          - Backprop + optimizer update
          - Return (loss, mean_reward) across the entire batch
        """
        if not batch_data:
            return 0.0, 0.0  # Nothing to train on

        def batch_closure(model_instance):
            total_loss = 0.0
            valid_count = 0

            for data in batch_data:
                loss_val = single_question_loss(
                    model=model_instance,
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

        # Compute forward & grad
        loss_value_and_grad = nn.value_and_grad(self.model, batch_closure)
        batch_loss, grads_dict = loss_value_and_grad(self.model)

        # Apply the gradients
        mx.eval(grads_dict)
        self.optimizer.update(self.model, grads_dict)

        # Compute final batch reward
        all_rewards = []
        for d in batch_data:
            all_rewards.extend(d["rewards"])
        final_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

        # Return the final batch loss & reward
        return float(batch_loss), final_reward

    # -----------------------------------------------------------------------
    # Example HOOK Overrides (Optional)
    # -----------------------------------------------------------------------
    def on_batch_end(self, epoch, batch_idx, loss, reward):
        """
        Called automatically by generic_train after each batch is processed.
        We'll log a summary here.
        """
        if self.verbose:
            logger.info(color_text(
                f"[GRPO MLX] E{epoch}B{batch_idx} => Loss: {loss:.4f}, Mean Reward: {reward:.4f}",
                YELLOW
            ))
        # You could also do other side effects here if needed.

    # If you'd like a final message at the end of training:
    # def on_train_end(self, mean_loss, mean_reward):
    #     logger.info(f"Training finished with final mean_loss={mean_loss:.4f}, mean_reward={mean_reward:.4f}")
