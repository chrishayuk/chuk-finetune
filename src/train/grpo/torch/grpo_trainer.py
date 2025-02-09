import logging
import torch
import numpy as np

from train.trainer_base import Trainer
from train.grpo.torch.grpo_utils import gather_kl_divergence, gather_logprobs
from train.grpo.torch.grpo_loss import compute_advantages, grpo_loss
from train.grpo.torch.grpo_generation import generate_single_response_and_oldlogprob
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

def ensure_dict_torch(item):
    """
    Torch-specific or general logic that tries to parse raw_item into { "prompt": ... }.
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
    GRPO Trainer for Torch, replicating MLX's approach with a shared prepare_batch_data function.
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

    def prepare_batch_data(self, batch_questions):
        """
        We call the shared function `prepare_batch_data_for_grpo`, providing:
          - ensure_dict_torch
          - a "single-response" generator lambda that calls generate_single_response_and_oldlogprob.
          - self.calculate_reward, self.G, self.verbose
        """
        # define a small local or lambda function that calls your Torch generation utility
        def generate_fn(prompt, verbose):
            # returns (resp_text, sum_lp)
            return generate_single_response_and_oldlogprob(
                model=self.model,
                tokenizer=self.tokenizer,
                prompt=prompt,
                verbose=verbose
            )
        
        # prepare the batch data
        batch_data = prepare_batch_data_for_grpo(
            batch_questions=batch_questions,
            ensure_dict_fn=ensure_dict_torch,
            generate_single_fn=generate_fn,
            calculate_reward=self.calculate_reward,
            G=self.G,
            verbose=self.verbose
        )

        # return the batch data
        return batch_data

    def train_step(self, batch_data):
        """
        For each item => re-compute new logprobs & KL, do GRPO loss => single backward
        """
        if not batch_data:
            return 0.0, 0.0

        total_loss = 0.0
        valid_count = 0
        all_rewards = []

        self.optimizer.zero_grad()

        for data_item in batch_data:
            responses = data_item["responses"]
            old_logprobs = data_item["old_logprobs"]
            rewards = data_item["rewards"]
            all_rewards.extend(rewards)

            if not responses:
                continue

            # 1) Compute advantage
            adv_arr = compute_advantages(rewards)  # shape [G]
            current_logprobs = []
            kl_values = []

            # 2) Recompute new logprobs & KL for each response
            for resp in responses:
                tokenized_resp = self.tokenizer(resp, return_tensors="pt")
                resp_inputs = {k: v.to(self.device) for k, v in tokenized_resp.items()}

                out_current = self.model(**resp_inputs)
                current_lp = gather_logprobs(out_current.logits, resp_inputs["input_ids"])
                current_logprobs.append(current_lp)

                with torch.no_grad():
                    out_ref = self.ref_model(**resp_inputs)
                    kl_val = gather_kl_divergence(
                        out_current.logits, out_ref.logits, resp_inputs["input_ids"]
                    )
                kl_values.append(kl_val)

            logprobs_current_sums = torch.cat(current_logprobs, dim=0)
            kl_sums = torch.cat(kl_values, dim=0)
            old_sums_t = torch.tensor(old_logprobs, dtype=torch.float32, device=self.device)
            adv_t = torch.tensor(adv_arr, dtype=torch.float32, device=self.device)

            # 3) GRPO loss
            item_loss = grpo_loss(
                logprobs_current=logprobs_current_sums,
                logprobs_old=old_sums_t,
                advantages=adv_t,
                kl_divergences=kl_sums,
                kl_coeff=self.kl_coeff
            )

            total_loss += item_loss
            valid_count += 1

        if valid_count > 0:
            total_loss /= valid_count

        total_loss.backward()
        self.optimizer.step()

        mean_loss = float(total_loss.item())
        import numpy as np
        mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

        return mean_loss, mean_reward

    def on_batch_end(self, epoch, batch_idx, loss, reward):
        logger.info(color_text(
            f"[Torch GRPO] E{epoch}B{batch_idx} => Loss: {loss:.4f}, Mean Reward: {reward:.4f}",
            YELLOW
        ))