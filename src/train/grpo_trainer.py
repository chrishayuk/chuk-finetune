# grpo_trainer.py
import torch
import numpy as np
from torch.utils.data import DataLoader
from torch.optim import AdamW

# imports
from train.grpo_utils import gather_logprobs, gather_kl_divergence
from train.grpo_loss import compute_advantages, grpo_loss

def train_step(
    base_model,
    ref_model,
    tokenizer,
    batch_questions,
    verifier,
    G,
    optimiser,
    calculate_reward,
    device,
    verbose=False
):
    """
    Single training step over a batch of questions:
      1. Generate G responses per question using the base model (via sampling).
      2. Compute old log-probs (treated as constants).
      3. Compute rewards + advantages.
      4. Compute current log-probs & KL (with gradient).
      5. Compute GRPO loss and step the optimiser.
    """
    losses = []

    for i, q in enumerate(batch_questions):
        if verbose:
            print(f"\n--- Batch item {i} ---")
            print(f"Question: {q}")

        # -------------------------------------------------
        # 1) Generate G responses via sampling
        # -------------------------------------------------
        inputs = tokenizer(q, return_tensors="pt").to(device)
        outputs = base_model.generate(
            **inputs,
            num_return_sequences=G,
            max_length=200,
            do_sample=True,           # <--- Turn on sampling
            top_p=0.95,               # <--- nucleus sampling
            top_k=50,                 # <--- or you could remove / tweak these
            temperature=0.7           # <--- sampling temperature
        )
        responses = [tokenizer.decode(output.cpu(), skip_special_tokens=True) for output in outputs]

        if verbose:
            print(f"Generated responses ({G} total):")
            for idx, resp in enumerate(responses):
                print(f"  Response {idx+1}: {resp}")

        # -------------------------------------------------
        # 2) Old log-probs as constants
        # -------------------------------------------------
        logprobs_old_sums = []
        with torch.no_grad():
            for response in responses:
                input_ids = tokenizer(response, return_tensors="pt").input_ids.to(device)
                # 'old' policy => no labels
                out = base_model(input_ids)
                sum_logprob = gather_logprobs(out.logits, input_ids)
                # store as float => no gradient
                logprobs_old_sums.append(sum_logprob.item())

        # -------------------------------------------------
        # 3) Rewards + advantages
        # -------------------------------------------------
        rewards = [calculate_reward(r, verifier) for r in responses]
        advantages_arr = compute_advantages(rewards)

        if verbose:
            print("Rewards:", rewards)
            print("Advantages:", advantages_arr)

        # -------------------------------------------------
        # 4) Current log-probs + KL (with gradient)
        # -------------------------------------------------
        current_logprob_list = []
        kl_list = []

        for response in responses:
            input_ids = tokenizer(response, return_tensors="pt").input_ids.to(device)

            # Current policy => no labels
            out_current = base_model(input_ids)
            sum_current = gather_logprobs(out_current.logits, input_ids)

            # Reference model => no gradient
            with torch.no_grad():
                out_ref = ref_model(input_ids)
            kl_val = gather_kl_divergence(out_current.logits, out_ref.logits, input_ids)

            current_logprob_list.append(sum_current)
            kl_list.append(kl_val)

        # Convert to shape [G]
        logprobs_current_sums = torch.cat(current_logprob_list, dim=0)
        kl_sums = torch.cat(kl_list, dim=0)

        # Convert old log-probs & advantages to device
        logprobs_old_sums = torch.tensor(logprobs_old_sums, dtype=torch.float32, device=device)
        advantages_t = torch.tensor(advantages_arr, dtype=torch.float32, device=device)

        # -------------------------------------------------
        # 5) Compute GRPO loss, backprop, step
        # -------------------------------------------------
        loss = grpo_loss(logprobs_current_sums, logprobs_old_sums, advantages_t, kl_sums)
        optimiser.zero_grad()
        loss.backward()
        optimiser.step()

        losses.append(loss.item())

        if verbose:
            print(f"Loss for this item: {loss.item():.4f}")

    return np.mean(losses)


def train_grpo(
    base_model,
    ref_model,
    tokenizer,
    verifier,
    dataset,
    calculate_reward,
    device=None,
    epochs=1,
    batch_size=4,
    G=4,
    lr=1e-5,
    verbose=False
):
    """
    High-level training loop for GRPO.
    """
    # load the dataset
    data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # setup the optimizer
    optimiser = AdamW(base_model.parameters(), lr=lr)

    # put the base model in training mode
    base_model.train()
    if verbose:
        print("\n[INFO] Starting training...")

    # perform the epochs
    for epoch in range(epochs):
        if verbose:
            print(f"\n[INFO] Epoch {epoch+1}/{epochs} in progress...")

        epoch_losses = []
        for batch_index, batch_questions in enumerate(data_loader):
            batch_questions = [str(q) for q in batch_questions]

            loss = train_step(
                base_model=base_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                batch_questions=batch_questions,
                verifier=verifier,
                G=G,
                optimiser=optimiser,
                calculate_reward=calculate_reward,
                device=device,
                verbose=verbose
            )
            epoch_losses.append(loss)

            if verbose:
                print(f"\nBatch {batch_index}, Mean loss: {loss:.4f}")
                print("-" * 40)

        avg_loss = np.mean(epoch_losses)
        print(f"[INFO] Epoch {epoch+1}/{epochs}, Mean loss: {avg_loss:.4f}")

    print("[INFO] GRPO training complete!")
