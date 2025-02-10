# src/train/grpo/torch/grpo_trainer.py

import logging
import torch
import numpy as np

# Core imports
from train.trainer_base import Trainer
from train.grpo.advantage_utils import compute_advantages
from train.grpo.grpo_prepare import prepare_batch_data_for_grpo
from train.grpo.torch.grpo_utils import gather_kl_divergence, gather_logprobs
from train.grpo.torch.grpo_loss import grpo_loss
from train.grpo.torch.grpo_generation import generate_single_response_and_oldlogprob

# Logging setup
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

def ensure_dict_torch(item):
    """
    Torch-specific or general logic that tries to parse raw_item into { "prompt": ... }.
    If it's a valid dict, return it directly.
    If it's a JSON-like string, parse it.
    Otherwise, if it's just a prompt string, wrap in { "prompt": prompt }.
    """
    if isinstance(item, dict):
        return item
    if isinstance(item, str) and item.strip().startswith("{"):
        import ast
        try:
            possible_dict = ast.literal_eval(item)
            if isinstance(possible_dict, dict):
                return possible_dict
        except:
            pass
    if isinstance(item, str):
        txt = item.strip().lower()
        if "prompt =" in txt or txt == "prompt":
            return None
        return {"prompt": item}
    return None

class GRPOTrainer(Trainer):
    """
    GRPO Trainer for Torch. Uses a single forward pass across the batch for efficiency,
    leveraging the shared 'prepare_batch_data_for_grpo' utility.

    'self.model' is the *current* model (being updated).
    'self.ref_model' is the *frozen* or old policy used to generate samples and 
    compute KL reference.
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

        # Set the device
        if self.device is None:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.ref_model.to(self.device)

    def prepare_batch_data(self, batch_questions):
        """
        Calls the shared 'prepare_batch_data_for_grpo' to handle:
          - Data item dict creation (ensure_dict_torch)
          - Single-response generation from the REF model (frozen policy)
          - Reward calculation and storing old log-probs
        """
        def generate_fn(prompt, verbose):
            # *** Using ref_model to generate => sampling from the old/frozen policy. ***
            return generate_single_response_and_oldlogprob(
                ref_model=self.ref_model,      # <<--- KEY CHANGE
                tokenizer=self.tokenizer,
                prompt=prompt,
                verbose=verbose
            )
        
        batch_data = prepare_batch_data_for_grpo(
            batch_questions=batch_questions,
            ensure_dict_fn=ensure_dict_torch,
            generate_single_fn=generate_fn,
            calculate_reward=self.calculate_reward,
            G=self.G,
            verbose=self.verbose
        )
        return batch_data
    
    def train_step(self, batch_data):
        if not batch_data:
            return 0.0, 0.0

        # ---------------------------
        # Flatten data
        # ---------------------------
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

        # ---------------------------
        # 1) Compute normalized advantages (NumPy -> Torch)
        # ---------------------------
        advantages_np = compute_advantages(all_rewards)
        advantages = torch.tensor(advantages_np, dtype=torch.float32, device=self.device)

        # ---------------------------
        # 2) Tokenize all responses at once
        # ---------------------------
        tokenized = self.tokenizer(
            all_responses,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}

        # ---------------------------
        # 3) Forward pass on current model
        # ---------------------------
        outputs_current = self.model(**tokenized)

        # ---------------------------
        # 4) Forward pass on reference model (no grad)
        # ---------------------------
        with torch.no_grad():
            outputs_ref = self.ref_model(**tokenized)

        # ---------------------------
        # 5) Gather logprobs & KL
        # ---------------------------
        logprobs_current = gather_logprobs(outputs_current.logits, tokenized["input_ids"])
        kl_values = gather_kl_divergence(
            outputs_current.logits,
            outputs_ref.logits,
            tokenized["input_ids"]
        )

        # ---------------------------
        # 6) Convert old logprobs to torch
        # ---------------------------
        old_logprobs_t = torch.tensor(all_old_logprobs, dtype=torch.float32, device=self.device)

        # ---------------------------
        # 7) Compute GRPO loss
        # ---------------------------
        total_loss = grpo_loss(
            logprobs_current=logprobs_current,
            logprobs_old=old_logprobs_t,
            advantages=advantages,
            kl_divergences=kl_values,
            kl_coeff=self.kl_coeff
        )

        # ---------------------------
        # 8) Backprop & step
        # ---------------------------
        self.optimizer.zero_grad()
        total_loss.backward()
        self.optimizer.step()

        # ---------------------------
        # 9) Calculate stats
        # ---------------------------
        mean_loss = float(total_loss.item())
        mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

        return mean_loss, mean_reward

    def on_batch_end(self, epoch, batch_idx, loss, reward):
        logger.info(color_text(
            f"[Torch GRPO] E{epoch}B{batch_idx} => Loss: {loss:.4f}, Mean Reward: {reward:.4f}",
            YELLOW
        ))
