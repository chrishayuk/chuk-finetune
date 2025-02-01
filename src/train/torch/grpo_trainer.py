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
    Single training step over one batch of questions.
    """
    losses = []

    for i, question in enumerate(batch_questions):
        if verbose:
            print(f"\n--- Torch GRPO step, item {i} ---")
            print(f"Question: {question}")

        inputs = tokenizer(question, return_tensors="pt")
        inputs = {k: v.to(device) for k, v in inputs.items()}

        outputs = base_model.generate(
            **inputs,
            num_return_sequences=G,
            max_length=200,
            do_sample=True,
            top_p=0.95,
            top_k=50,
            temperature=0.7
        )
        responses = [tokenizer.decode(o.cpu(), skip_special_tokens=True) for o in outputs]

        # Print out the generated responses
        if verbose:
            print("Generated responses:")
            for idx, resp in enumerate(responses, start=1):
                print(f"  Response {idx}: {resp}")

        # 2) Compute old log-probs
        logprobs_old_sums = []
        with torch.no_grad():
            for resp in responses:
                resp_inputs = tokenizer(resp, return_tensors="pt")
                resp_inputs = {k: v.to(device) for k, v in resp_inputs.items()}
                out = base_model(**resp_inputs)
                sum_logprob = gather_logprobs(out.logits, resp_inputs["input_ids"])
                logprobs_old_sums.append(sum_logprob.item())

        # 3) Rewards + advantages
        rewards = [calculate_reward(r, verifier) for r in responses]
        advantages_arr = compute_advantages(rewards)
        if verbose:
            print(f"Rewards: {rewards}")
            print(f"Advantages: {advantages_arr}")

        # 4) Current log-probs + KL
        current_list = []
        kl_list = []
        for resp in responses:
            resp_inputs = tokenizer(resp, return_tensors="pt")
            resp_inputs = {k: v.to(device) for k, v in resp_inputs.items()}

            out_current = base_model(**resp_inputs)
            sum_current = gather_logprobs(out_current.logits, resp_inputs["input_ids"])

            with torch.no_grad():
                out_ref = ref_model(**resp_inputs)

            kl_val = gather_kl_divergence(
                out_current.logits,
                out_ref.logits,
                resp_inputs["input_ids"]
            )

            current_list.append(sum_current)
            kl_list.append(kl_val)

        logprobs_current_sums = torch.cat(current_list, dim=0)
        kl_sums = torch.cat(kl_list, dim=0)
        old_sums_t = torch.tensor(logprobs_old_sums, dtype=torch.float32, device=device)
        adv_t = torch.tensor(advantages_arr, dtype=torch.float32, device=device)

        # 5) Compute GRPO loss, backprop, step optimizer
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


def train_grpo(
    base_model,
    ref_model,
    tokenizer,
    verifier,
    data_loader,         # This is your Torch DataLoader
    calculate_reward,
    optimizer,
    epochs: int = 1,
    batch_size: int = 4,
    G: int = 4,
    device=None,
    verbose=False
):
    """
    High-level Torch GRPO training function. Iterates over 'epochs',
    loops over data_loader, calls train_step(...) for each batch,
    and returns the mean loss across all epochs.
    """
    all_epoch_losses = []

    for epoch in range(epochs):
        if verbose:
            print(f"[Torch] Starting epoch {epoch+1}/{epochs}...")
        epoch_losses = []

        for batch_questions in data_loader:
            # The 'batch_questions' should be a list/tuple of strings from your DataLoader
            batch_loss = train_step(
                base_model=base_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                batch_questions=batch_questions,
                verifier=verifier,
                G=G,
                optimizer=optimizer,
                calculate_reward=calculate_reward,
                device=device,
                verbose=verbose
            )
            epoch_losses.append(batch_loss)

        avg_loss = np.mean(epoch_losses)
        all_epoch_losses.append(avg_loss)
        if verbose:
            print(f"[Torch] Epoch {epoch+1} -> Mean loss: {avg_loss:.4f}")

    return np.mean(all_epoch_losses)
