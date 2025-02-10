import torch

def gather_logprobs(logits: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
    """
    Gathers the log probabilities for each token in `input_ids`, then sums over the sequence dimension.
    Expects:
        - logits: [B, T, V] (batch_size, seq_len, vocab_size)
        - input_ids: [B, T]
    Returns:
        A 1D tensor of shape [B], each element is the sum of log probs for that sequence.
    """
    # log-softmax over vocab dimension => shape [B, T, V]
    logprobs = logits.log_softmax(dim=-1)

    # Gather the log-prob for the specific token at each position => shape [B, T]
    gathered = torch.gather(
        logprobs,
        dim=-1,
        index=input_ids.unsqueeze(-1)  # shape [B, T, 1]
    ).squeeze(-1)  # => [B, T]

    # Sum over sequence length => shape [B]
    seq_logprob = gathered.sum(dim=-1)
    return seq_logprob


def gather_kl_divergence(
    current_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    input_ids: torch.Tensor
) -> torch.Tensor:
    """
    Computes the KL divergence between the current distribution and the reference distribution,
    only for the tokens specified by `input_ids`, and sums over the sequence dimension.
    Expects:
        - current_logits: [B, T, V]
        - ref_logits: [B, T, V]
        - input_ids: [B, T]
    Returns:
        A 1D tensor of shape [B], each element is the sum of per-token KL for that sequence:
            KL = sum( p(x) * [log p(x) - log q(x)] ) over tokens in the sequence.
    """
    # log-softmax over vocab => [B, T, V]
    current_logprobs = current_logits.log_softmax(dim=-1)
    ref_logprobs = ref_logits.log_softmax(dim=-1)

    # Gather log-probs for the specific token => [B, T]
    gathered_current = torch.gather(
        current_logprobs,
        dim=-1,
        index=input_ids.unsqueeze(-1)
    ).squeeze(-1)

    gathered_ref = torch.gather(
        ref_logprobs,
        dim=-1,
        index=input_ids.unsqueeze(-1)
    ).squeeze(-1)

    # p(x) = exp(current_logprobs)
    p_x = gathered_current.exp()  # => [B, T]

    # KL token-wise = p(x) * (log p(x) - log q(x)) => [B, T]
    kl_tokens = p_x * (gathered_current - gathered_ref)

    # Sum over the sequence length => shape [B]
    kl_val = kl_tokens.sum(dim=-1)
    return kl_val
