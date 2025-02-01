# src/train/torch/grpo_trainer.py
import torch
import numpy as np

from train.torch.grpo_utils import gather_logprobs, gather_kl_divergence
from train.torch.grpo_loss import compute_advantages, grpo_loss

def train_step(
    base_model,
    ref_model,
    tokenizer,
    batch_questions,
    verifier,
    G,
    optimizer,
    calculate_reward,
    device,
    verbose=False
):
    """
    Single training step using Torch-based logic:

      1) Generate G responses per question by sampling from base_model.
      2) Compute old log-probs (treated as constants).
      3) Compute rewards and advantages (normalised).
      4) Compute current log-probs and KL divergence (with gradient).
      5) Compute the GRPO loss, backpropagate, and step the optimizer.
    """
    losses = []

    for i, question in enumerate(batch_questions):
        if verbose:
            print(f"\n--- Torch GRPO step, item {i} ---")
            print(f"Question: {question}")

        # -------------------------------------------------
        # 1) Generate G responses by sampling
        # -------------------------------------------------
        # 1a) Tokenise
        inputs = tokenizer(question, return_tensors="pt")
        # 1b) Move each tensor in 'inputs' to device
        inputs = {k: v.to(device) for k, v in inputs.items()}

        # Now we can call base_model.generate(...)
        outputs = base_model.generate(
            **inputs,
            num_return_sequences=G,
            max_length=200,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7
        )
        # Convert each sequence to text
        responses = [tokenizer.decode(o.cpu(), skip_special_tokens=True) for o in outputs]

        # -------------------------------------------------
        # 2) Compute old log-probs (constants, no gradient)
        # -------------------------------------------------
        logprobs_old_sums = []
        with torch.no_grad():
            for resp in responses:
                # Tokenise the response
                resp_inputs = tokenizer(resp, return_tensors="pt")
                resp_inputs = {k: v.to(device) for k, v in resp_inputs.items()}

                out = base_model(**resp_inputs)  # old policy
                sum_logprob = gather_logprobs(out.logits, resp_inputs["input_ids"])
                logprobs_old_sums.append(sum_logprob.item())

        # -------------------------------------------------
        # 3) Rewards + advantages
        # -------------------------------------------------
        rewards = [calculate_reward(r, verifier) for r in responses]
        advantages_arr = compute_advantages(rewards)  # returns a NumPy array
        if verbose:
            print(f"Rewards: {rewards}")
            print(f"Advantages: {advantages_arr}")

        # -------------------------------------------------
        # 4) Current log-probs + KL (with gradient)
        # -------------------------------------------------
        current_list = []
        kl_list = []
        for resp in responses:
            # tokenise response
            resp_inputs = tokenizer(resp, return_tensors="pt")
            resp_inputs = {k: v.to(device) for k, v in resp_inputs.items()}

            out_current = base_model(**resp_inputs)
            sum_current = gather_logprobs(out_current.logits, resp_inputs["input_ids"])

            with torch.no_grad():
                out_ref = ref_model(**resp_inputs)
            kl_val = gather_kl_divergence(out_current.logits, out_ref.logits, resp_inputs["input_ids"])

            current_list.append(sum_current)
            kl_list.append(kl_val)

        # Convert to PyTorch tensors
        logprobs_current_sums = torch.cat(current_list, dim=0)
        kl_sums = torch.cat(kl_list, dim=0)
        old_sums_t = torch.tensor(logprobs_old_sums, dtype=torch.float32, device=device)
        adv_t = torch.tensor(advantages_arr, dtype=torch.float32, device=device)

        # -------------------------------------------------
        # 5) Compute GRPO loss, backprop, and optimizer step
        # -------------------------------------------------
        loss_val = grpo_loss(
            logprobs_current=logprobs_current_sums,
            logprobs_old=old_sums_t,
            advantages=adv_t,
            kl_divergences=kl_sums
        )
        optimizer.zero_grad()
        loss_val.backward()
        optimizer.step()

        losses.append(loss_val.item())
        if verbose:
            print(f"Loss (item {i}): {loss_val.item():.4f}")

    return np.mean(losses)
