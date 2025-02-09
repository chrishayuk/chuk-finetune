# src/inference/torch/samplers/top_p_with_kvcache.py

import torch
import torch.nn.functional as F

from inference.stop_utils import prepare_stop_sequences, check_stop_sequences
from inference.prompt_removal import remove_prompt_prefix

def top_p_generate_torch_with_kvcache(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 200,
    temperature: float = 1.0,
    top_p: float = 0.95,
    stop_sequences=None,
    remove_prompt: bool = True,
    device=None,
):
    """
    Token-by-token top-p (nucleus) sampling in Torch, using a KV-cache 
    (past_key_values) to avoid re-running the entire sequence each time.

    We fix the 'nan/inf' error by always keeping at least the highest-probability token.
    Steps:
      1) Encode the entire prompt and feed it once to fill `past_key_values`.
      2) For each new token, feed only that token + `past_key_values`
         (the model will skip re-processing old tokens).
      3) For top-p filtering in descending order, we 'shift' the cutoff mask so 
         the top token is never masked, thus avoiding an all-zero distribution.
    """
    # Prepare stop sequences
    if stop_sequences is None:
        stop_sequences = []
    stop_seqs = prepare_stop_sequences(stop_sequences)

    # Encode prompt
    tokens = tokenizer.encode(prompt)
    if not tokens:
        tokens = [tokenizer.eos_token_id] if tokenizer.eos_token_id is not None else []
        if not tokens:
            raise ValueError("No tokens found and no eos_token_id in tokenizer.")

    # Decide on device
    if device is None:
        if hasattr(model, "device"):
            device = model.device
        else:
            device = torch.device("cpu")

    # Convert prompt -> tensor on device
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    # 1) First forward pass: feed entire prompt
    #    We expect outputs = model(..., use_cache=True) => .logits, .past_key_values
    with torch.no_grad():
        outputs = model(input_ids, use_cache=True)
        logits = outputs.logits  # shape [1, seq_len, vocab_size]
        past_key_values = outputs.past_key_values  # The KV cache

    # We'll generate up to max_new_tokens
    generated_tokens = tokens[:]  # copy the prompt IDs

    # 2) Iterative decoding
    for _ in range(max_new_tokens):
        # a) Take logits for the last token
        last_logits = logits[:, -1, :]  # shape [1, vocab_size]

        # b) Temperature
        if temperature != 1.0:
            last_logits = last_logits / temperature

        # c) Convert logits -> probabilities
        probs = F.softmax(last_logits, dim=-1)  # shape [1, vocab_size]

        # d) Sort descending => top-p
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # e) Identify the tokens that exceed top_p
        cutoff_mask = cumulative_probs > top_p

        # ---- SHIFT the mask by 1 so we never zero out the highest-prob token ----
        # (This ensures the distribution never becomes all-zero or sums to zero.)
        cutoff_mask = torch.roll(cutoff_mask, 1, dims=-1)
        cutoff_mask[0, 0] = False  # keep the top token

        # f) Zero out everything after the cutoff
        sorted_probs[cutoff_mask] = 0.0

        # g) Re-normalize
        sum_trunc = sorted_probs.sum(dim=-1, keepdim=True) + 1e-12
        sorted_probs = sorted_probs / sum_trunc

        # h) Sample from the truncated distribution
        sample_idx_in_sorted = torch.multinomial(sorted_probs, 1).item()
        next_token_id = sorted_indices[0, sample_idx_in_sorted].item()

        # i) Append to sequence
        generated_tokens.append(next_token_id)

        # Check for EOS
        if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
            break

        # Check custom stop sequences
        current_text = tokenizer.decode(generated_tokens)
        maybe_stopped = check_stop_sequences(current_text, stop_seqs)
        if maybe_stopped is not None:
            current_text = maybe_stopped
            break

        # j) Next step: feed only the new token + the existing KV-cache
        next_input_ids = torch.tensor([[next_token_id]], device=device)
        with torch.no_grad():
            outputs = model(next_input_ids, use_cache=True, past_key_values=past_key_values)
            logits = outputs.logits  # shape [1, 1, vocab_size]
            past_key_values = outputs.past_key_values

    # Done
    raw_output = tokenizer.decode(generated_tokens)
    if remove_prompt:
        return remove_prompt_prefix(raw_output, prompt)
    return raw_output