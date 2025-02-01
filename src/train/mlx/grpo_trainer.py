# src/train/mlx/grpo_trainer.py

import numpy as np
import mlx.core as mx
from mlx_lm import generate
from train.mlx.grpo_loss import compute_advantages, grpo_loss
from train.mlx.grpo_utils import gather_logprobs, gather_kl_divergence

def train_step(
    base_model,
    ref_model,
    tokenizer,
    batch_questions,
    verifier,
    G,
    optimizer,
    device=None,
    calculate_reward=None,
    verbose=False
):
    """
    Single training step using MLX-based logic for sampling, log-probs, KL, and GRPO loss.

    1) Generate G responses per question (via mlx_lm.generate).
    2) Compute old log-probs (treated as constants).
    3) Compute rewards and normalised advantages.
    4) Compute current log-probs & KL divergence.
    5) Compute GRPO loss.
    6) Build and evaluate gradient dictionary.
    7) Update the model.
    8) Return mean loss over all questions in the batch.
    """
    losses = []

    for i, question in enumerate(batch_questions):
        if verbose:
            print(f"\n--- MLX GRPO step, item {i} ---")
            print(f"Question: {question}")

        # -------------------------------------------------
        # 1) Generate G responses from the base_model
        # -------------------------------------------------
        responses = []
        for _ in range(G):
            response_text = generate(
                model=base_model,
                tokenizer=tokenizer,
                prompt=question,
                max_tokens=200,
                verbose=False
            )
            responses.append(response_text.strip())

        # -------------------------------------------------
        # 2) Compute old log-probs
        # -------------------------------------------------
        logprobs_old_sums = []
        for resp in responses:
            old_input_tokens = tokenizer.encode(resp)
            old_logits = base_model(mx.array(old_input_tokens, mx.uint32)[None])
            sum_lp = gather_logprobs(old_logits, old_input_tokens)
            logprobs_old_sums.append(float(sum_lp))

        # -------------------------------------------------
        # 3) Rewards + advantages
        # -------------------------------------------------
        rewards = [calculate_reward(r, verifier) for r in responses]
        advantages_arr = compute_advantages(rewards)
        if verbose:
            print("Rewards:", rewards)
            print("Advantages:", advantages_arr)

        # -------------------------------------------------
        # 4) Current log-probs + KL
        # -------------------------------------------------
        current_list = []
        kl_list = []
        for resp in responses:
            tokens = tokenizer.encode(resp)
            out_current = base_model(mx.array(tokens, mx.uint32)[None])
            sum_current = gather_logprobs(out_current, tokens)

            out_ref = ref_model(mx.array(tokens, mx.uint32)[None])
            kl_val = gather_kl_divergence(out_current, out_ref, tokens)

            current_list.append(sum_current)
            kl_list.append(kl_val)

        logprobs_current_sums = mx.concat(current_list, axis=0)
        kl_sums = mx.concat(kl_list, axis=0)
        old_sums_m = mx.array(logprobs_old_sums)
        advantages_m = mx.array(advantages_arr)

        # -------------------------------------------------
        # 5) Compute GRPO loss
        # -------------------------------------------------
        loss_val = grpo_loss(
            logprobs_current=logprobs_current_sums,
            logprobs_old=old_sums_m,
            advantages=advantages_m,
            kl_divergences=kl_sums
        )

        # -------------------------------------------------
        # 6) Build gradient dict from the model's parameters
        # -------------------------------------------------
        params_dict = base_model.parameters()  # e.g. {"param": mx.array(...)}
        grad_dict = mx.grad(params_dict, loss_val)

        # -------------------------------------------------
        # 7) Evaluate each gradient (convert from lazy transforms to real arrays)
        # -------------------------------------------------
        for k, v in grad_dict.items():
            grad_dict[k] = mx.eval(v)

        # -------------------------------------------------
        # 8) Update the model
        # -------------------------------------------------
        optimizer.update(base_model, grad_dict)

        losses.append(float(loss_val))
        if verbose:
            print(f"Loss (item {i}): {float(loss_val):.4f}")

    return np.mean(losses)
