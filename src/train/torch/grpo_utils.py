# src/train/torch/grpo_utils.py
import torch

def gather_logprobs(logits, input_ids):
    """
    Torch version: gather log-prob of each token, sum.
    Expects logits shape [1, seq_len, vocab_size], input_ids shape [seq_len].
    Returns shape [1].
    """
    # 1) log softmax
    logprobs = logits.log_softmax(dim=-1)  # shape [1, seq_len, vocab_size]

    # 2) Ensure input_ids has batch dim => [1, seq_len]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)  # shape [1, seq_len]

    # 3) gather => shape [1, seq_len]
    gathered = torch.gather(
        logprobs,
        dim=-1,
        index=input_ids.unsqueeze(-1)  # shape [1, seq_len, 1]
    ).squeeze(-1)  # => shape [1, seq_len]

    # 4) sum across seq_len => shape [1]
    seq_logprob = gathered.sum(dim=1)
    return seq_logprob  # shape [1]

def gather_kl_divergence(current_logits, ref_logits, input_ids):
    """
    Torch version: gather KL = sum( p(x) * [log p(x) - log q(x)] ) across tokens in input_ids.
    Expects both logits [1, seq_len, vocab_size].
    Returns shape [1].
    """
    current_logprobs = current_logits.log_softmax(dim=-1)  # [1, seq_len, vocab]
    ref_logprobs     = ref_logits.log_softmax(dim=-1)

    # Expand input_ids => [1, seq_len]
    if input_ids.dim() == 1:
        input_ids = input_ids.unsqueeze(0)

    # Gather current log-probs => shape [1, seq_len]
    gathered_current = torch.gather(
        current_logprobs,
        dim=-1,
        index=input_ids.unsqueeze(-1)
    ).squeeze(-1)

    # Gather reference log-probs => shape [1, seq_len]
    gathered_ref = torch.gather(
        ref_logprobs,
        dim=-1,
        index=input_ids.unsqueeze(-1)
    ).squeeze(-1)

    # p(x) = exp(current)
    p_x = gathered_current.exp()  # shape [1, seq_len]

    # tokens => p(x)*[log p(x) - log q(x)]
    kl_tokens = p_x * (gathered_current - gathered_ref)
    kl_val = kl_tokens.sum(dim=1)  # shape [1]

    return kl_val
