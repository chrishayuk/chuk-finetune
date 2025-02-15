# src/train/grpo/torch/grpo_utils.py
import torch
import logging

logger = logging.getLogger(__name__)

def gather_logprobs(
    logits: torch.Tensor,
    input_ids: torch.Tensor,
    mask: torch.Tensor = None,
    chunk_size: int = 1000
) -> torch.Tensor:
    """
    Gathers the log probabilities for each token in `input_ids` by computing log_softmax in chunks.
    
    Optionally applies a mask (of shape [B, T]) so that tokens with mask==0 are ignored.
    
    Expects:
        - logits: [B, T, V] (batch_size, seq_len, vocab_size)
        - input_ids: [B, T]
        - mask: [B, T] (optional; 1 for valid tokens, 0 for tokens to ignore)
    
    Returns:
        A 1D tensor of shape [B], where each element is the sum of log probabilities for that sequence (only for valid tokens).
    """
    B, T, V = logits.shape
    logprobs_chunks = []
    logger.debug(f"Starting gather_logprobs: logits shape {logits.shape}, input_ids shape {input_ids.shape}")
    
    # Process the logits in chunks to avoid memory issues.
    for start in range(0, T, chunk_size):
        end = min(start + chunk_size, T)
        logits_chunk = logits[:, start:end, :]
        logprobs_chunk = torch.nn.functional.log_softmax(logits_chunk, dim=-1)
        logprobs_chunks.append(logprobs_chunk)
        logger.debug(f"Processed chunk from {start} to {end}: chunk shape {logits_chunk.shape}")
    
    # Concatenate the chunks along the sequence dimension.
    logprobs = torch.cat(logprobs_chunks, dim=1)
    # Gather log probabilities corresponding to the tokens in input_ids.
    gathered = torch.gather(logprobs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    if mask is not None:
        # Zero out contributions from tokens where mask == 0.
        gathered = gathered * mask
    # Sum over the sequence dimension.
    seq_logprob = gathered.sum(dim=-1)
    logger.debug(f"Final gathered log probabilities shape: {seq_logprob.shape}")
    return seq_logprob

def gather_kl_divergence(
    current_logits: torch.Tensor,
    ref_logits: torch.Tensor,
    input_ids: torch.Tensor,
    mask: torch.Tensor = None
) -> torch.Tensor:
    """
    Computes the KL divergence between the current distribution and the reference distribution
    for the tokens specified by `input_ids`, summed over the sequence dimension.
    
    Optionally applies a mask (of shape [B, T]) so that tokens with mask==0 are ignored.
    
    Expects:
        - current_logits: [B, T, V]
        - ref_logits: [B, T, V]
        - input_ids: [B, T]
        - mask: [B, T] (optional)
    
    Returns:
        A 1D tensor of shape [B], where each element is the sum of per-token KL divergences for that sequence.
    """
    logger.debug(f"Starting gather_kl_divergence: current_logits shape {current_logits.shape}, ref_logits shape {ref_logits.shape}")
    
    # Compute log-softmax for both distributions.
    current_logprobs = current_logits.log_softmax(dim=-1)
    ref_logprobs = ref_logits.log_softmax(dim=-1)
    
    # Gather log probabilities for the tokens specified by input_ids.
    gathered_current = torch.gather(current_logprobs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    gathered_ref = torch.gather(ref_logprobs, dim=-1, index=input_ids.unsqueeze(-1)).squeeze(-1)
    
    if mask is not None:
        gathered_current = gathered_current * mask
        gathered_ref = gathered_ref * mask
    
    # Convert log probabilities to probabilities.
    p_x = gathered_current.exp()
    
    # Compute per-token KL divergence.
    kl_tokens = p_x * (gathered_current - gathered_ref)

    if mask is not None:
        kl_val = (kl_tokens * mask).sum(dim=-1)
    else:
        kl_val = kl_tokens.sum(dim=-1)
    
    logger.debug(f"KL divergence per sequence shape: {kl_val.shape}")
    return kl_val
