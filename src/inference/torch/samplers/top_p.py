# src/inference/torch/samplers/top_p.py

import torch
import torch.nn.functional as F

from inference.stop_utils import prepare_stop_sequences, check_stop_sequences
from inference.prompt_removal import remove_prompt_prefix

def top_p_generate_torch(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2000,
    temperature: float = 0.6,
    top_p: float = 0.95,
    stop_sequences=None,
    remove_prompt: bool = True
):
    """
    Token-by-token Top-p (nucleus) sampling in Torch, 
    with optional stop sequences and prompt-prefix removal.

    NOTE: This code still re-runs the entire sequence each step 
          (no KV-cache). For large models, consider an incremental 
          decode approach.

    We fix potential "zero distribution" or NaN/Inf issues by:
      1) Clamping logits to avoid huge overflow in softmax
      2) Shifting the cutoff mask so at least the highest-prob token is always kept
      3) If the truncated sum < 1e-12, fallback to top token
      4) Clamping any NaNs/negatives in probabilities 
    """
    # 1) Prepare stop sequences
    stop_seqs = prepare_stop_sequences(stop_sequences)

    # 2) Basic setup
    eos_id = tokenizer.eos_token_id
    tokens = tokenizer.encode(prompt)
    if not tokens:
        if eos_id is None:
            raise ValueError("No tokens found and no eos_token_id in tokenizer.")
        tokens = [eos_id]

    # Attempt to find model device, else default CPU
    device = getattr(model, "device", torch.device("cpu"))

    # 3) Generate tokens
    for _ in range(max_new_tokens):
        # Convert tokens -> input_ids on correct device
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # Forward pass (no grad)
        with torch.no_grad():
            # shape: [batch=1, seq_len, vocab_size]
            logits = model(input_ids).logits

        # 4) Get logits for the last token
        next_token_logits = logits[:, -1, :]

        # 5) Temperature scaling
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # --- NEW: clip logits to avoid overflow in softmax ---
        next_token_logits = torch.clamp(next_token_logits, min=-100.0, max=100.0)

        # 6) Convert logits -> probabilities
        probs = F.softmax(next_token_logits, dim=-1)[0]  # shape [vocab_size]

        # 7) Sort descending for top-p
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 8) Identify tokens that exceed top_p
        cutoff_mask = cumulative_probs > top_p

        # SHIFT the cutoff mask by 1 so the top token isn't zeroed out
        cutoff_mask = torch.roll(cutoff_mask, 1, dims=-1)
        cutoff_mask[0] = False
        sorted_probs[cutoff_mask] = 0.0

        # --- clamp any potential NaNs or negatives ---
        sorted_probs = torch.nan_to_num(sorted_probs, nan=0.0, posinf=0.0, neginf=0.0)
        sorted_probs = torch.clamp(sorted_probs, min=0.0, max=1.0)

        # 9) Re-normalize or fallback
        sum_trunc = sorted_probs.sum()
        if sum_trunc < 1e-12:
            # fallback to picking the single highest-prob token
            next_token_id = sorted_indices[0].item()
        else:
            sorted_probs = sorted_probs / sum_trunc
            sample_idx_in_sorted = torch.multinomial(sorted_probs, 1).item()
            next_token_id = sorted_indices[sample_idx_in_sorted].item()

        # 10) Append new token
        tokens.append(next_token_id)

        # Early stop: EOS
        if eos_id is not None and next_token_id == eos_id:
            break

        # Custom stop sequences
        current_text = tokenizer.decode(tokens)
        maybe_truncated = check_stop_sequences(current_text, stop_seqs)
        if maybe_truncated is not None:
            return maybe_truncated

    # 11) Final decode + optional prompt removal
    raw_output = tokenizer.decode(tokens)
    if remove_prompt:
        return remove_prompt_prefix(raw_output, prompt)
    return raw_output
