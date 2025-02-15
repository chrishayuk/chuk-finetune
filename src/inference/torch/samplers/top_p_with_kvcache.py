# src/inference/torch/samplers/top_p_with_kvcache.py

import torch
import torch.nn.functional as F

from inference.stop_utils import prepare_stop_sequences, check_stop_sequences
from inference.prompt_removal import remove_prompt_prefix

def top_p_generate_torch_with_kvcache(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 512,
    temperature: float = 1.0,
    top_p: float = 0.95,
    stop_sequences=None,
    remove_prompt: bool = True,
    device=None,
):
    """
    A robust top-p (nucleus) sampler in Torch with KV-cache, preventing NaN/Inf
    probabilities. We do:
      1) Clip logits to avoid overflow in softmax
      2) Shift the cutoff mask so the top token isn't zeroed out
      3) Clamp negative or NaN values to [0,1]
      4) Fallback to the top token if sum ~ 0
    """
    # 1) Prepare stop sequences
    if stop_sequences is None:
        stop_sequences = []
    stop_seqs = prepare_stop_sequences(stop_sequences)

    # 2) Encode the prompt
    tokens = tokenizer.encode(prompt)
    if not tokens:
        # If empty, fallback to an eos token
        if tokenizer.eos_token_id is not None:
            tokens = [tokenizer.eos_token_id]
        else:
            raise ValueError("No tokens found and no eos_token_id in tokenizer.")

    # 3) Decide on device
    if device is None:
        if hasattr(model, "device"):
            device = model.device
        else:
            device = torch.device("cpu")

    # 4) Convert prompt to device
    input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

    with torch.no_grad():
        # We expect outputs.logits and outputs.past_key_values
        outputs = model(input_ids, use_cache=True)
        logits = outputs.logits
        past_key_values = outputs.past_key_values

    # We'll store all tokens in generated_tokens
    generated_tokens = tokens[:]

    for _ in range(max_new_tokens):
        # a) Get logits for the last token
        last_logits = logits[:, -1, :]  # shape [1, vocab_size]

        # b) Temperature
        if temperature != 1.0:
            last_logits = last_logits / temperature

        # --- NEW: clip logits to avoid huge exp() in softmax ---
        # E.g., clamp to [-100, 100], which is usually safe.
        last_logits = torch.clamp(last_logits, min=-100.0, max=100.0)

        # c) Softmax
        probs = F.softmax(last_logits, dim=-1)[0]  # shape [vocab_size]

        # d) Sort descending
        sorted_probs, sorted_indices = torch.sort(probs, descending=True)
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # e) Identify cutoff
        cutoff_mask = (cumulative_probs > top_p)
        # SHIFT by 1 so we never remove the top token
        cutoff_mask = torch.roll(cutoff_mask, 1, dims=-1)
        cutoff_mask[0] = False
        sorted_probs[cutoff_mask] = 0.0

        # --- NEW: clamp negative or NaN just in case ---
        # (Rarely needed if the above code is correct, but we do it for safety.)
        sorted_probs = torch.nan_to_num(sorted_probs, nan=0.0, posinf=0.0, neginf=0.0)
        sorted_probs = torch.clamp(sorted_probs, min=0.0, max=1.0)

        # f) Sum -> re-normalize or fallback
        sum_trunc = sorted_probs.sum()
        if sum_trunc < 1e-12:
            # fallback: pick the single highest-prob token
            next_token_id = sorted_indices[0].item()
        else:
            sorted_probs = sorted_probs / sum_trunc
            sample_idx_in_sorted = torch.multinomial(sorted_probs, 1).item()
            next_token_id = sorted_indices[sample_idx_in_sorted].item()

        generated_tokens.append(next_token_id)

        # g) EOS check
        if tokenizer.eos_token_id is not None and next_token_id == tokenizer.eos_token_id:
            break

        # h) custom stop sequences
        current_text = tokenizer.decode(generated_tokens)
        maybe_stopped = check_stop_sequences(current_text, stop_seqs)
        if maybe_stopped is not None:
            current_text = maybe_stopped
            break

        # i) Next step: feed only the new token
        next_input_ids = torch.tensor([[next_token_id]], device=device)
        with torch.no_grad():
            outputs = model(next_input_ids, use_cache=True, past_key_values=past_key_values)
            logits = outputs.logits
            past_key_values = outputs.past_key_values

    # final decode
    raw_output = tokenizer.decode(generated_tokens)
    if remove_prompt:
        return remove_prompt_prefix(raw_output, prompt)
    return raw_output
