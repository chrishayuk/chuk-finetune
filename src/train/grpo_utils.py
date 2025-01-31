# src/train/grpo_utils.py
import torch

def gather_logprobs(logits, input_ids):
    """
    Given logits of shape [1, seq_len, vocab_size] and the corresponding token IDs,
    gather the log-prob of the *actual* token at each timestep and sum to get 1 scalar.
    
    Returns a scalar tensor of shape [1]: sum of log-probs for the entire sequence.
    """
    logprobs = logits.log_softmax(dim=-1)  # [1, seq_len, vocab_size]
    
    gathered = torch.gather(
        logprobs,
        dim=-1,
        index=input_ids.unsqueeze(-1)  # shape [1, seq_len, 1]
    ).squeeze(-1)  # becomes [1, seq_len]
    
    seq_logprob = gathered.sum(dim=1)  # [1]
    return seq_logprob

def gather_kl_divergence(current_logits, ref_logits, input_ids):
    """
    Computes the KL divergence for a single response, approximating
    sum( p(x) * [log p(x) - log q(x)] ) over all tokens.
    Returns a scalar tensor of shape [1].
    """
    current_logprobs = current_logits.log_softmax(dim=-1)  # [1, seq_len, vocab_size]
    ref_logprobs = ref_logits.log_softmax(dim=-1)          # [1, seq_len, vocab_size]

    gathered_current = torch.gather(
        current_logprobs, 
        dim=-1,
        index=input_ids.unsqueeze(-1)
    ).squeeze(-1)  # [1, seq_len]
    
    gathered_ref = torch.gather(
        ref_logprobs,
        dim=-1,
        index=input_ids.unsqueeze(-1)
    ).squeeze(-1)  # [1, seq_len]

    current_probs = gathered_current.exp()  # [1, seq_len]
    kl_tokens = current_probs * (gathered_current - gathered_ref)  # [1, seq_len]
    kl_val = kl_tokens.sum(dim=1)  # [1]

    return kl_val
