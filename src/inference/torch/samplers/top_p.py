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
    
    NOTE: This code still re-runs the entire sequence each step. 
          For large models, implement KV-caching to speed it up.
    """
    # Prepare stop sequences
    stop_seqs = prepare_stop_sequences(stop_sequences)

    # Basic init
    eos_id = tokenizer.eos_token_id
    tokens = tokenizer.encode(prompt)
    if not tokens:
        if eos_id is None:
            raise ValueError("No tokens found and no eos_token_id in tokenizer.")
        tokens = [eos_id]

    device = getattr(model, "device", torch.device("cpu"))

    for _ in range(max_new_tokens):
        # Convert tokens -> input_ids on correct device
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        # Forward pass (no grad)
        with torch.no_grad():
            # model(...) => [batch, seq_len, vocab_size]
            logits = model(input_ids).logits

        # Get logits for the last token in the sequence
        next_token_logits = logits[:, -1, :]  # shape [1, vocab_size]

        # Temperature scaling
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Convert logits -> probabilities
        # shape: [1, vocab_size], so we take [0] to get a 1D tensor of shape [vocab_size]
        probs = F.softmax(next_token_logits, dim=-1)[0]

        # ---- Top-p (nucleus) filtering in descending order ----
        # 1. Sort tokens by descending probability
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # 2. Identify the tokens that push us past top_p
        cutoff_mask = cumulative_probs > top_p

        # 3. Zero out everything past the cutoff
        #    (We keep at least the first token even if it’s > top_p itself,
        #     but in normal conditions we rarely see a single prob > top_p.)
        # A quick trick: shift cutoff_mask right by 1 so the “first True” remains included
        cutoff_mask = torch.roll(cutoff_mask, 1, dims=-1)
        cutoff_mask[0] = False  # The highest-prob token is always kept
        sorted_probs[cutoff_mask] = 0.0

        # 4. Re-normalize to sum to 1
        sorted_probs = sorted_probs / (sorted_probs.sum(dim=-1, keepdim=True) + 1e-12)

        # 5. Sample from the truncated + renormalized distribution
        #    (Use torch.multinomial on the re-scaled probabilities)
        sample_idx_in_sorted = torch.multinomial(sorted_probs, 1).item()
        next_token_id = sorted_indices[sample_idx_in_sorted].item()

        # Append the new token
        tokens.append(next_token_id)

        # Stop if we hit EOS
        if eos_id is not None and next_token_id == eos_id:
            break

        # Check custom stop sequences
        current_text = tokenizer.decode(tokens)
        maybe_truncated = check_stop_sequences(current_text, stop_seqs)
        if maybe_truncated is not None:
            return maybe_truncated

    # Decode the entire generated text
    raw_output = tokenizer.decode(tokens)

    # Optionally remove the original prompt from the final output
    if remove_prompt:
        return remove_prompt_prefix(raw_output, prompt)
    return raw_output

