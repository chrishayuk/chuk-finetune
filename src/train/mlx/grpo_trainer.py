import numpy as np
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import generate

from train.mlx.grpo_loss import compute_advantages, grpo_loss
from train.mlx.grpo_utils import gather_logprobs, gather_kl_divergence


def generate_responses_and_oldlogprobs(
    model,
    tokenizer,
    question,
    G,
    verbose=False
):
    """
    Generates G responses (untraced), computes old log-probs, 
    and optionally prints the responses if verbose=True.
    """
    responses = []
    logprobs_old_sums = []

    for idx in range(G):
        response_text = generate(
            model=model,
            tokenizer=tokenizer,
            prompt=question,
            max_tokens=200,
            verbose=False
        )
        resp = response_text.strip()
        responses.append(resp)

        if verbose:
            print(f"  Generated response {idx+1}: {resp}")

        old_tokens = tokenizer.encode(resp)
        old_logits = model(mx.array(old_tokens, mx.uint32)[None])
        sum_lp = gather_logprobs(old_logits, old_tokens)
        logprobs_old_sums.append(float(sum_lp))

    return responses, logprobs_old_sums


def compute_grpo_loss(
    model,
    ref_model,
    tokenizer,
    question: str,
    G: int,
    verifier,
    calculate_reward,
    responses,
    old_logprobs,
    verbose=False
):
    """
    Given precomputed 'responses' and 'old_logprobs', compute:
      - Rewards + advantages
      - Current log-probs + KL (traced by autograd)
      - Final GRPO loss
    Now logs rewards/advantages if verbose=True.
    """
    # 1) Rewards + advantages
    rewards = [calculate_reward(r, verifier) for r in responses]
    advantages_arr = compute_advantages(rewards)
    if verbose:
        print(f"Rewards: {rewards}")
        print(f"Advantages: {advantages_arr}")

    # 2) Current log-probs & KL (tracked by autograd)
    current_list = []
    kl_list = []
    for resp in responses:
        tokens = tokenizer.encode(resp)
        out_current = model(mx.array(tokens, mx.uint32)[None])
        sum_current = gather_logprobs(out_current, tokens)

        out_ref = ref_model(mx.array(tokens, mx.uint32)[None])
        kl_val = gather_kl_divergence(out_current, out_ref, tokens)

        current_list.append(sum_current)
        kl_list.append(kl_val)

    logprobs_current_sums = mx.concat(current_list, axis=0)
    kl_sums               = mx.concat(kl_list, axis=0)

    old_sums_m   = mx.array(old_logprobs)
    advantages_m = mx.array(advantages_arr)

    # 3) Final GRPO loss
    loss_val = grpo_loss(
        logprobs_current=logprobs_current_sums,
        logprobs_old=old_sums_m,
        advantages=advantages_m,
        kl_divergences=kl_sums
    )
    return loss_val


def single_question_loss(
    model,
    ref_model,
    tokenizer,
    question,
    G,
    verifier,
    calculate_reward,
    responses,
    old_logprobs,
    verbose=False
):
    """
    Minimal function wrapped by value_and_grad(...). 
    Only does final log-probs, KL, and GRPO loss computations (with autograd).
    """
    return compute_grpo_loss(
        model=model,
        ref_model=ref_model,
        tokenizer=tokenizer,
        question=question,
        G=G,
        verifier=verifier,
        calculate_reward=calculate_reward,
        responses=responses,
        old_logprobs=old_logprobs,
        verbose=verbose
    )


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
    Single MLX training step over a batch:
      - Generate responses + old log-probs outside autograd
      - Compute GRPO loss inside a value_and_grad closure
      - Update the model
      - Print responses/rewards if verbose=True
    """
    losses = []

    def closure(model, question, responses, old_logprobs):
        return single_question_loss(
            model=model,
            ref_model=ref_model,
            tokenizer=tokenizer,
            question=question,
            G=G,
            verifier=verifier,
            calculate_reward=calculate_reward,
            responses=responses,
            old_logprobs=old_logprobs,
            verbose=verbose
        )

    loss_value_and_grad = nn.value_and_grad(base_model, closure)

    for i, question in enumerate(batch_questions):
        if verbose:
            print(f"\n--- MLX GRPO step, item {i} ---")
            print(f"Question: {question}")

        # Generate + old-logprobs (untraced)
        responses, old_logprobs = generate_responses_and_oldlogprobs(
            model=base_model,
            tokenizer=tokenizer,
            question=question,
            G=G,
            verbose=verbose
        )

        # Compute (loss, grads) with autograd
        loss_val, grads_dict = loss_value_and_grad(
            base_model,
            question,
            responses,
            old_logprobs
        )
        mx.eval(grads_dict)
        optimizer.update(base_model, grads_dict)

        final_loss = float(loss_val)
        if verbose:
            print(f"Loss (item {i}): {final_loss:.4f}")

        losses.append(final_loss)

    return np.mean(losses)


def train_grpo(
    base_model,
    ref_model,
    tokenizer,
    verifier,
    data_iterator,        # yields batches of questions
    calculate_reward,
    optimizer,
    epochs: int = 1,
    batch_size: int = 4,
    G: int = 4,
    device=None,
    verbose=False
):
    """
    High-level loop:
      - For each epoch, re-instantiate data_iterator
      - For each batch, call train_step(...)
      - Return mean loss
    """
    all_epoch_losses = []

    for epoch in range(epochs):
        if verbose:
            print(f"[MLX] Starting epoch {epoch+1}/{epochs}...")
        epoch_losses = []

        for batch_questions in data_iterator():
            loss = train_step(
                base_model=base_model,
                ref_model=ref_model,
                tokenizer=tokenizer,
                batch_questions=batch_questions,
                verifier=verifier,
                G=G,
                optimizer=optimizer,
                device=device,
                calculate_reward=calculate_reward,
                verbose=verbose
            )
            epoch_losses.append(loss)

        avg_loss = np.mean(epoch_losses)
        all_epoch_losses.append(avg_loss)
        if verbose:
            print(f"[MLX] Epoch {epoch+1} -> Mean loss: {avg_loss:.4f}")

    return np.mean(all_epoch_losses)
