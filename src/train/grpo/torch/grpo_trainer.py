# src/train/grpo/torch/grpo_trainer.py
import torch
import numpy as np

# imports
from train.trainer_base import Trainer
from train.grpo.torch.grpo_utils import gather_logprobs, gather_kl_divergence
from train.grpo.torch.grpo_loss import compute_advantages, grpo_loss

def ensure_dict(item):
    """
    Parallels the MLX ensure_dict(...) utility.
    Ensures the input is a dict (or attempts to parse if it's a JSON-like string).
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
    # If no success, assume the prompt is just a raw string
    return {"prompt": item}


class TorchGRPOTrainer(Trainer):
    """
    A class-based GRPO Trainer for Torch, mirroring the MLX version
    but using PyTorch operators and logic.
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
        self.device = device if device is not None else torch.device("cpu")
        self.verbose = verbose

    def prepare_batch_data(self, batch_questions):
        """
        Receives a list of items (each might be a dict or raw string).
        For each item:
          - Generate G responses
          - Compute "old" log-probs (from the same model used for generation)
          - Compute rewards
          - If reward == None, skip that item
        Returns a list of dicts with:
          {
            "item": original_item_dict,
            "responses": [...],
            "old_logprobs": [...],
            "rewards": [...]
          }
        """
        batch_data = []

        for i, raw_item in enumerate(batch_questions):
            item = ensure_dict(raw_item)
            prompt = item["prompt"].strip()

            if self.verbose:
                print(f"\n=== Prompt {i} ===")
                print(prompt)

            # 1) Generate G responses
            # Instead of .to(self.device) on the entire dict, map its contents
            tokenized = self.tokenizer(prompt, return_tensors="pt")
            inputs = {k: v.to(self.device) for k, v in tokenized.items()}
            outputs = self.model.generate(
                **inputs,
                num_return_sequences=self.G,
                max_length=128,
                do_sample=True,
                top_p=0.95,
                top_k=50,
                temperature=0.7
            )
            responses = [
                self.tokenizer.decode(o.cpu(), skip_special_tokens=True)
                for o in outputs
            ]

            old_logprobs = []
            rewards_list = []
            skip_this_item = False

            # 2) Compute old log-probs & rewards
            with torch.no_grad():
                for resp_idx, resp in enumerate(responses):
                    tokenized_resp = self.tokenizer(resp, return_tensors="pt")
                    resp_inputs = {k: v.to(self.device) for k, v in tokenized_resp.items()}
                    
                    out = self.model(**resp_inputs)
                    sum_logprob = gather_logprobs(out.logits, resp_inputs["input_ids"])
                    old_logprobs.append(sum_logprob.item())

                    # Reward function expects: (resp, item)
                    score, feedback_text = self.calculate_reward(resp, item)
                    if score is None:
                        if self.verbose:
                            print(f"[SKIP] No reward for response {resp_idx}: '{resp}'")
                        skip_this_item = True
                        break
                    rewards_list.append(score)

            if skip_this_item:
                continue

            # 3) Accumulate in batch_data
            batch_data.append({
                "item": item,
                "responses": responses,
                "old_logprobs": old_logprobs,
                "rewards": rewards_list,
            })

        return batch_data

    def train_step(self, batch_data):
        """
        Aggregates the GRPO loss across items in batch_data:
          - For each item, re-computes new log-probs & KL vs ref_model
          - Calculates advantage from item["rewards"]
          - Averages losses across items, single backward + optimizer step
          Returns (loss, mean_reward) as floats.
        """
        if not batch_data:
            return 0.0, 0.0

        total_loss = 0.0
        valid_count = 0
        all_rewards = []

        # Zero-grad once before accumulating
        self.optimizer.zero_grad()

        for i, data_item in enumerate(batch_data):
            responses = data_item["responses"]
            old_logprobs = data_item["old_logprobs"]
            rewards = data_item["rewards"]
            all_rewards.extend(rewards)

            if not responses:
                continue

            # 1) Compute advantage
            advantages_arr = compute_advantages(rewards)  # shape [G]

            # 2) Recompute new log-probs & KL
            current_logprobs = []
            kl_values = []

            for resp in responses:
                tokenized_resp = self.tokenizer(resp, return_tensors="pt")
                resp_inputs = {k: v.to(self.device) for k, v in tokenized_resp.items()}

                out_current = self.model(**resp_inputs)
                sum_logprob = gather_logprobs(out_current.logits, resp_inputs["input_ids"])
                current_logprobs.append(sum_logprob)

                with torch.no_grad():
                    out_ref = self.ref_model(**resp_inputs)
                kl_val = gather_kl_divergence(
                    out_current.logits,
                    out_ref.logits,
                    resp_inputs["input_ids"]
                )
                kl_values.append(kl_val)

            # Convert lists to Tensors
            logprobs_current_sums = torch.cat(current_logprobs, dim=0)  # shape [G]
            kl_sums = torch.cat(kl_values, dim=0)                      # shape [G]
            old_sums_t = torch.tensor(old_logprobs, dtype=torch.float32, device=self.device)
            adv_t = torch.tensor(advantages_arr, dtype=torch.float32, device=self.device)

            # 3) Compute GRPO loss for this item
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
            total_loss = total_loss / valid_count

        total_loss.backward()
        self.optimizer.step()

        mean_loss = float(total_loss.item())
        mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

        if self.verbose:
            print(f"[Torch GRPO] Batch Update => Loss: {mean_loss:.4f}, Reward: {mean_reward:.4f}")

        return mean_loss, mean_reward
