# src/train/grpo/mlx/grpo_trainer.py
import ast
import logging
import numpy as np
import mlx.core as mx
import mlx.nn as nn

# imports
from train.trainer_base import Trainer
from train.grpo.mlx.grpo_loss import single_question_loss
from train.grpo.mlx.grpo_generation import generate_single_response_and_oldlogprob
from train.grpo.grpo_prepare import prepare_batch_data_for_grpo

# Logging setup
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Optional color codes (for nicer console logs)
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"

def color_text(text, color):
    return f"{color}{text}{RESET}"

def ensure_dict_mlx(item):
    """
    Attempt to ensure 'item' is a dict. If it's already a dict, return it.
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
    GRPO Trainer for MLX framework, now using the shared function
    'prepare_batch_data_for_grpo' to build the 'batch_data'.
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
        # Call parent constructor
        super().__init__(model, tokenizer, optimizer, device=device, verbose=verbose)

        # Set the reference model, reward calculation, etc.
        self.ref_model = ref_model
        self.calculate_reward = calculate_reward
        self.G = G
        self.kl_coeff = kl_coeff

    def prepare_batch_data(self, batch_questions):
        """
        Use the shared function, injecting MLX-specific 'ensure_dict_mlx'
        and the single-response generator 'generate_single_response_and_oldlogprob'.
        """
        def generate_single_fn(prompt, vb):
            return generate_single_response_and_oldlogprob(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                verbose=vb
            )
        
        # Prepare the batch data
        batch_data = prepare_batch_data_for_grpo(
            batch_questions=batch_questions,
            ensure_dict_fn=ensure_dict_mlx,
            generate_single_fn=generate_single_fn,
            calculate_reward=self.calculate_reward,
            G=self.G,
            verbose=self.verbose
        )
        return batch_data

    def train_step(self, batch_data):
        """
        Perform a GRPO update with single_question_loss across batch_data.
        """
        if not batch_data:
            # Nothing to train on
            return 0.0, 0.0  

        def batch_closure(model_instance):
            total_loss = 0.0
            valid_count = 0

            # Loop through each item in the batch
            for data in batch_data:
                # Calculate loss for this question
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
        
        # Compute loss & gradients
        loss_value_and_grad = nn.value_and_grad(self.model, batch_closure)
        batch_loss, grads_dict = loss_value_and_grad(self.model)

        # Evaluate gradient, then optimize
        mx.eval(grads_dict)
        self.optimizer.update(self.model, grads_dict)

        # Compute final batch reward
        all_rewards = []
        for d in batch_data:
            all_rewards.extend(d["rewards"])
        final_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

        # Log batch info
        logger.info(color_text(
            f"\n[GRPO MLX] Single Batch Update => Loss: {batch_loss:.4f}, "
            f"Mean Reward: {final_reward:.4f}\n",
            YELLOW
        ))

        return float(batch_loss), final_reward

    def on_batch_end(self, epoch, batch_idx, loss, reward):
        logger.info(color_text(
            f"[GRPO MLX] E{epoch}B{batch_idx} => Loss: {loss:.4f}, Mean Reward: {reward:.4f}",
            YELLOW
        ))
