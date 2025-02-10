# src/train/grpo/mlx/grpo_trainer.py
import ast
import logging
import numpy as np
import mlx.core as mx
import mlx.nn as nn

# train imports
from train.trainer_base import Trainer

# generic grpo imports
from train.grpo.advantage_utils import compute_advantages
from train.grpo.grpo_prepare import prepare_batch_data_for_grpo

# mlx specific grpo imports
from train.grpo.mlx.grpo_utils import gather_logprobs, gather_kl_divergence
from train.grpo.mlx.grpo_loss import grpo_loss
from train.grpo.mlx.grpo_generation import generate_single_response_and_oldlogprob

logger = logging.getLogger(__name__)

# Optional color codes
RESET = "\033[0m"
BOLD = "\033[1m"
GREEN = "\033[92m"
YELLOW = "\033[93m"
CYAN = "\033[96m"
RED = "\033[91m"

def color_text(text, color):
    return f"{color}{text}{RESET}"

def ensure_dict_mlx(item):
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
    MLX-based GRPO Trainer that does a single forward pass per batch,
    mirroring the Torch approach.
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
        super().__init__(model, tokenizer, optimizer, device=device, verbose=verbose)
        self.ref_model = ref_model
        self.calculate_reward = calculate_reward
        self.G = G
        self.kl_coeff = kl_coeff

    def prepare_batch_data(self, batch_questions):
        """
        Shared data prep: parse each item => generate => store old_logprobs & rewards
        """
        def generate_single_fn(prompt, vb):
            return generate_single_response_and_oldlogprob(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                verbose=vb
            )
        
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
        Single-pass training step: flatten all responses from 'batch_data',
        do one forward pass on current model, one on reference model, 
        compute GRPO loss, and apply gradients.
        """
        if not batch_data:
            return 0.0, 0.0
        
        # 1) Flatten data
        all_responses = []
        all_old_logprobs = []
        all_rewards = []

        for data_item in batch_data:
            responses = data_item["responses"]
            old_logprobs = data_item["old_logprobs"]
            rewards = data_item["rewards"]

            if not responses:
                continue

            all_responses.extend(responses)
            all_old_logprobs.extend(old_logprobs)
            all_rewards.extend(rewards)

        if not all_responses:
            return 0.0, 0.0

        # 2) Compute advantages (NumPy => MLX array)
        advantages_arr = compute_advantages(all_rewards)  # shape [N] np.float32
        advantages_m = mx.array(advantages_arr, mx.float32)

        # 3) Tokenize all responses at once (we assume your tokenizer can handle batch -> B,seq)
        #    Because MLX might not have an HF-like "batch encode" method, we do it manually:
        #    Convert each response to tokens, store in a list-of-lists
        all_token_ids = []
        max_len = 0
        for resp in all_responses:
            tokens = self.tokenizer.encode(resp)
            if not tokens:
                tokens = [self.tokenizer.eos_token_id]
            max_len = max(max_len, len(tokens))
            all_token_ids.append(tokens)

        # Now build an int32 array [B, max_len], pad with eos or 0
        B = len(all_token_ids)
        input_ids_np = np.zeros((B, max_len), dtype=np.uint32)  # or eos
        eos_id = self.tokenizer.eos_token_id
        for i, tokens in enumerate(all_token_ids):
            length = len(tokens)
            input_ids_np[i, :length] = tokens

        # MLX array => shape [B, max_len]
        input_ids_m = mx.array(input_ids_np, mx.uint32)

        # 4) Forward pass on current model => logits shape [B, seq_len, vocab_size]
        def batch_closure(model_instance):
            # We'll compute the entire batch's logits
            out_current = model_instance(input_ids_m)
            # gather logprobs => shape [B]
            logprobs_current = gather_logprobs(out_current, input_ids_m)

            # forward ref (no grad) => gather kl => shape [B]
            out_ref = self.ref_model(input_ids_m)
            kl_values = gather_kl_divergence(out_current, out_ref, input_ids_m)

            # convert old_logprobs => MLX
            old_logprobs_m = mx.array(all_old_logprobs, mx.float32)

            # 5) Compute GRPO loss
            loss_val = grpo_loss(
                logprobs_current=logprobs_current,
                logprobs_old=old_logprobs_m,
                advantages=advantages_m,
                kl_divergences=kl_values,
                clip_range=0.2,
                kl_coeff=self.kl_coeff,
                reduction="mean"
            )
            return loss_val

        # 5) value_and_grad => compute loss + grads
        loss_value_and_grad = nn.value_and_grad(self.model, batch_closure)
        batch_loss, grads_dict = loss_value_and_grad(self.model)

        # 6) Apply gradients with MLX optimizer
        mx.eval(grads_dict)
        self.optimizer.update(self.model, grads_dict)

        # 7) Stats
        mean_loss = float(batch_loss)
        mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

        # Logging
        logger.info(color_text(
            f"\n[GRPO MLX] Single Batch Update => Loss: {mean_loss:.4f}, Mean Reward: {mean_reward:.4f}\n",
            YELLOW
        ))

        return mean_loss, mean_reward

    def on_batch_end(self, epoch, batch_idx, loss, reward):
        logger.info(color_text(
            f"[GRPO MLX] E{epoch}B{batch_idx} => Loss: {loss:.4f}, Mean Reward: {reward:.4f}",
            YELLOW
        ))