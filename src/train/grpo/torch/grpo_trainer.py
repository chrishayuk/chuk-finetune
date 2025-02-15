# src/train/grpo/torch/grpo_trainer.py
import logging
import os
import torch
import numpy as np
from src.device.torch_device_memory import log_device_memory
from src.device.tensor_debug import debug_tensor_info  # Import our tensor debug helper

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
    Parses raw item into a dictionary with a "prompt" key.
    """
    if isinstance(item, dict):
        return item
    if isinstance(item, str) and item.strip().startswith("{"):
        import ast
        try:
            possible_dict = ast.literal_eval(item)
            if isinstance(possible_dict, dict):
                return possible_dict
        except Exception as e:
            logger.debug(f"Parsing error: {e}")
    if isinstance(item, str):
        txt = item.strip().lower()
        if "prompt =" in txt or txt == "prompt":
            return None
        return {"prompt": item}
    return None

class GRPOTrainer(Trainer):
    """
    GRPO Trainer for Torch.

    Implements the Group Relative Policy Optimization (GRPO) objective as described in DeepSeek‑R1‑Zero.
    For each question, a group of outputs {o₁, …, o_G} is sampled from the old (frozen) policy.
    The advantage for each output is computed relative to the group:
    
         Aᵢ = (rᵢ - mean({r₁,…,r_G})) / std({r₁,…,r_G})
    
    The loss objective is:
    
         L = - E[min(r * A_norm, clip(r, 1-ε, 1+ε) * A_norm)] + β * KL
    
    where r = exp(logπ_new - logπ_old), ε is the clip range, and β (kl_coeff) is the KL penalty weight.
    
    This class also uses an adaptive sample mask to skip degenerate samples and tracks cumulative loss per batch.

    Now includes forward hooks to detect which module triggers NaNs/Infs.
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

        if self.device is None:
            self.device = torch.device("cpu")
        self.model.to(self.device)
        self.ref_model.to(self.device)
        
        # Initialize cumulative loss (reset externally at the start of each epoch).
        self.cumulative_loss = 0.0

        # Register forward hooks to track down where NaNs or Infs occur.
        self._register_nan_hooks(self.model, name_prefix="CurrentModel")
        self._register_nan_hooks(self.ref_model, name_prefix="RefModel")
        
        # Log initial memory state.
        log_device_memory(tag="GRPOTrainer __init__", device=str(self.device))

    def _register_nan_hooks(self, root_module, name_prefix=""):
        """
        Registers forward hooks on every submodule to detect when outputs contain NaNs or Infs.
        This helps pinpoint exactly which layer is causing the numerical explosion.
        """
        def nan_hook(module, inputs, output):
            if output is None:
                # Some layers might return None for e.g. attention weights
                return
            
            # If output is a single tensor
            if isinstance(output, torch.Tensor):
                if torch.isnan(output).any() or torch.isinf(output).any():
                    logger.error(
                        f"[NaN/Inf Hook] Detected in {name_prefix}.{module.__class__.__name__}. "
                        "Output contains NaNs or Infs!"
                    )
            # If the module returns a tuple/list (some Transformer blocks return (hidden_states, attn_weights) or similar)
            elif isinstance(output, (tuple, list)):
                for i, out_i in enumerate(output):
                    if out_i is None:
                        # Sometimes attention weights are optional or the module returns (hidden_states, None).
                        continue
                    if not isinstance(out_i, torch.Tensor):
                        # If it's not a tensor, skip it
                        continue
                    if torch.isnan(out_i).any() or torch.isinf(out_i).any():
                        logger.error(
                            f"[NaN/Inf Hook] Detected in {name_prefix}.{module.__class__.__name__} "
                            f"(output idx={i}). Output contains NaNs or Infs!"
                        )
            # else: skip anything that is not a tensor, tuple, or list

        for name, submodule in root_module.named_modules():
            submodule.register_forward_hook(nan_hook)


    def prepare_batch_data(self, batch_questions):
        """
        Prepares batch data by:
          - Converting raw items into dictionaries (via ensure_dict_torch)
          - Generating a group of responses using the frozen reference model
          - Computing rewards and storing the old log-probabilities.
        """
        def generate_fn(prompt, verbose):
            return generate_single_response_and_oldlogprob(
                ref_model=self.ref_model,
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

        # Flatten the batch data.
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

        # Possibly store the batch data for debugging if NaNs appear
        # (especially if you want to log or skip).
        self.current_batch_text = all_responses
        self.current_batch_rewards = all_rewards

        # 1) Compute normalized advantages.
        advantages_np = compute_advantages(all_rewards)
        advantages = torch.tensor(advantages_np, dtype=torch.float32, device=self.device)
        debug_tensor_info(advantages, "advantages")
        log_device_memory(tag="After computing advantages", device=str(self.device))
        
        # 2) Tokenize all responses.
        tokenized = self.tokenizer(
            all_responses,
            return_tensors="pt",
            padding=True,
            truncation=True
        )
        tokenized = {k: v.to(self.device) for k, v in tokenized.items()}
        for key, tensor in tokenized.items():
            debug_tensor_info(tensor, f"tokenized[{key}]")
        log_device_memory(tag="After tokenization", device=str(self.device))

        # 3) Forward pass on the current model.
        outputs_current = self.model(**tokenized)
        outputs_current.logits = outputs_current.logits.float()
        # The forward hooks above will already log if any submodule produced NaNs.
        if torch.isnan(outputs_current.logits).any():
            logger.error("NaNs detected in outputs_current.logits at final. Replacing with -1e9.")
            logger.error(f"Batch text (for debugging): {self.current_batch_text}")
            logger.error(f"Batch rewards: {self.current_batch_rewards}")
            outputs_current.logits = torch.nan_to_num(outputs_current.logits, nan=-1e9)
        debug_tensor_info(outputs_current.logits, "outputs_current.logits")
        log_device_memory(tag="After forward pass (current model)", device=str(self.device))

        # 4) Forward pass on the reference model (no gradients).
        with torch.no_grad():
            outputs_ref = self.ref_model(**tokenized)
        outputs_ref.logits = outputs_ref.logits.float()
        if torch.isnan(outputs_ref.logits).any():
            logger.error("NaNs detected in outputs_ref.logits at final. Replacing with -1e9.")
            logger.error(f"Batch text (for debugging): {self.current_batch_text}")
            logger.error(f"Batch rewards: {self.current_batch_rewards}")
            outputs_ref.logits = torch.nan_to_num(outputs_ref.logits, nan=-1e9)
        debug_tensor_info(outputs_ref.logits, "outputs_ref.logits")
        log_device_memory(tag="After forward pass (reference model)", device=str(self.device))

        # 5) Gather log probabilities & compute KL divergence.
        logprobs_current = gather_logprobs(outputs_current.logits, tokenized["input_ids"])
        debug_tensor_info(logprobs_current, "logprobs_current")
        kl_values = gather_kl_divergence(
            outputs_current.logits,
            outputs_ref.logits,
            tokenized["input_ids"]
        )
        debug_tensor_info(kl_values, "kl_values")
        log_device_memory(tag="After gathering logprobs and KL", device=str(self.device))

        # 6) Convert old log-probabilities.
        old_logprobs_t = torch.tensor(all_old_logprobs, dtype=torch.float32, device=self.device)
        debug_tensor_info(old_logprobs_t, "old_logprobs_t")

        # 7) Compute an adaptive sample mask.
        if logprobs_current.dim() == 1:
            total_logprob = logprobs_current
        else:
            total_logprob = logprobs_current.sum(dim=1)  # shape: [B]

        logger.debug(f"Per-sample total log probabilities: {total_logprob.tolist()}")

        mean_logprob = total_logprob.mean()
        std_logprob = total_logprob.std()
        alpha = 0.5
        threshold = mean_logprob - alpha * std_logprob
        logger.debug(f"Adaptive threshold => mean={mean_logprob.item():.4f}, std={std_logprob.item():.4f}, threshold={threshold.item():.4f}")

        sample_mask = (total_logprob >= threshold).float()
        logger.debug(f"Computed sample mask => {sample_mask.tolist()}")

        if sample_mask.sum() < 1e-6:
            logger.warning("No valid samples in this batch => skipping update. (NaN or degenerate batch?)")
            return 0.0, 0.0

        # 8) Compute the GRPO loss using the sample mask.
        total_loss = grpo_loss(
            logprobs_current=logprobs_current,
            logprobs_old=old_logprobs_t,
            advantages=advantages,
            kl_divergences=kl_values,
            kl_coeff=self.kl_coeff,
            sample_mask=sample_mask
        )
        debug_tensor_info(total_loss, "total_loss")
        log_device_memory(tag="After loss computation", device=str(self.device))

        if sample_mask.sum() < 1e-6:
            logger.warning("No valid samples after grpo_loss => skipping update.")
            return 0.0, 0.0

        # 9) Backpropagation & optimizer step.
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        log_device_memory(tag="After backward", device=str(self.device))

        self.optimizer.step()
        log_device_memory(tag="After optimizer step", device=str(self.device))

        # 10) Compute statistics.
        mean_loss = float(total_loss.item())
        mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

        # Update cumulative loss
        if not hasattr(self, 'cumulative_loss'):
            self.cumulative_loss = 0.0
        self.cumulative_loss += mean_loss

        return mean_loss, mean_reward

    def on_batch_end(self, epoch, batch_idx, loss, reward):
        logger.info(color_text(
            f"[Torch GRPO] E{epoch}B{batch_idx} => Batch Loss: {loss:.4f}, "
            f"Cumulative Loss: {self.cumulative_loss:.4f}, Mean Reward: {reward:.4f}",
            YELLOW
        ))
