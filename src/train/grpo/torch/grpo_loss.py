import torch
import logging

logger = logging.getLogger(__name__)

def grpo_loss(
    logprobs_current: torch.Tensor,    # shape [B, T] or [B]
    logprobs_old: torch.Tensor,        # shape [B, T] or [B]
    advantages: torch.Tensor,          # shape [B, T] or [B]
    kl_divergences: torch.Tensor,      # shape [B, T] or [B]
    clip_range: float = 0.2,
    kl_coeff: float = 0.1,
    reduction: str = "mean",
    sample_mask: torch.Tensor = None,  # shape [B, T] or [B], optional
    prob_threshold: float = -50.0
) -> torch.Tensor:
    """
    Computes a GRPO/PPO-like loss with ratio clipping, advantage normalization, and optional
    per-sample skipping based on a log-prob threshold. It handles either token-level or
    sequence-level tensors, as long as shapes are consistent or broadcastable.

    Args:
        logprobs_current (Tensor): Current log-probs for either each token [B, T] or entire sequence [B].
        logprobs_old (Tensor): Old log-probs, same shape as `logprobs_current`.
        advantages (Tensor): Advantages for each token [B, T] or each sequence [B].
        kl_divergences (Tensor): KL divergences for each token [B, T] or each sequence [B].
        clip_range (float): PPO clipping range (e.g., 0.2).
        kl_coeff (float): Weighting for the KL term in the final loss.
        reduction (str): Either "mean", "sum", or "none" to indicate how to combine the final loss.
        sample_mask (Tensor, optional): A mask of shape [B, T] or [B]. 1.0 = valid, 0.0 = ignore.
        prob_threshold (float): If sample_mask is None, we compute a [B] mask by
                                summing logprobs_current along time and discarding sequences
                                with sum below this threshold.

    Returns:
        final_loss (Tensor): A scalar if `reduction` in ["mean","sum"], or a tensor shaped like
                             the unmasked loss if `reduction` == "none".
    """

    # -------------------------------------------------------------------------
    # 1) Convert log probabilities to float and replace NaNs/Infs with safe values
    # -------------------------------------------------------------------------
    logprobs_current = torch.nan_to_num(
        logprobs_current.float(),
        nan=-1e9, posinf=1e9, neginf=-1e9
    )
    logprobs_old = torch.nan_to_num(
        logprobs_old.float(),
        nan=-1e9, posinf=1e9, neginf=-1e9
    )

    # -------------------------------------------------------------------------
    # 2) Normalize advantages (always done at sequence level)
    #    We'll flatten to 1D for mean/std if the user passed [B, T].
    # -------------------------------------------------------------------------
    flat_adv = advantages.view(-1)  # flatten for mean/std
    adv_mean = flat_adv.mean()
    adv_std = flat_adv.std() + 1e-8
    advantages_norm = (advantages - adv_mean) / adv_std

    logger.debug(f"adv_mean: {adv_mean.item():.3f}, adv_std: {adv_std.item():.3f}")

    # -------------------------------------------------------------------------
    # 3) Compute ratio: r = exp( clamp(logπ_new - logπ_old, -5, 5) )
    # -------------------------------------------------------------------------
    diff = torch.clamp(logprobs_current - logprobs_old, min=-5.0, max=5.0)
    ratios = torch.exp(diff)

    # -------------------------------------------------------------------------
    # 4) Compute the clipped surrogate: - min(r * A, clamp(r,1-ε,1+ε)* A)
    #    We do negative because we're minimizing the negative of the PPO objective
    # -------------------------------------------------------------------------
    surr1 = ratios * advantages_norm
    surr2 = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * advantages_norm
    clipped_surrogate = -torch.min(surr1, surr2)

    # -------------------------------------------------------------------------
    # 5) Compute KL penalty: kl_coeff * kl_divergences
    #    Shapes can be [B, T] or [B]. Broadcasting must match clipped_surrogate
    # -------------------------------------------------------------------------
    # Ensure kl_divergences is float and broadcast if needed:
    kl_divergences = kl_divergences.float()
    # We simply rely on PyTorch broadcasting if kl_divergences is [B] and clipped_surrogate is [B, T].
    # If they're both [B, T], they match exactly.
    # If they're [B], repeated across T dimension automatically.
    kl_term = kl_coeff * kl_divergences

    # Combine
    loss = clipped_surrogate + kl_term

    # -------------------------------------------------------------------------
    # 6) Clamp final loss to avoid extreme explosion
    # -------------------------------------------------------------------------
    loss = torch.clamp(loss, -1e4, 1e4)

    # -------------------------------------------------------------------------
    # 7) If user did not provide sample_mask, create one from the sum of logprobs
    #    (only if the logprobs have a time dimension).
    # -------------------------------------------------------------------------
    if sample_mask is None:
        # If we have shape [B, T], sum across T. If [B], it remains as is.
        if logprobs_current.ndim == 2:
            total_logprob = logprobs_current.sum(dim=1)  # shape [B]
        else:
            total_logprob = logprobs_current  # shape [B] (already)
        sample_mask = (total_logprob > prob_threshold).float()  # shape [B]

    # Now we need to broadcast sample_mask to match `loss` if necessary.
    # If loss is [B, T] and sample_mask is [B], we expand dims to match.
    while sample_mask.ndim < loss.ndim:
        sample_mask = sample_mask.unsqueeze(-1)  # expand last dim

    # -------------------------------------------------------------------------
    # 8) Apply sample_mask and reduce
    # -------------------------------------------------------------------------
    masked_loss = loss * sample_mask  # zero out invalid tokens/samples

    if reduction == "mean":
        # sum over all tokens, then divide by the number of "valid" tokens
        denom = sample_mask.sum()
        final_loss = masked_loss.sum() / (denom + 1e-8)
    elif reduction == "sum":
        final_loss = masked_loss.sum()
    else:
        # 'none' => return the full loss tensor, shape [B, T] or [B]
        final_loss = masked_loss

    return final_loss