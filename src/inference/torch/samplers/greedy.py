# src/inference/torch/samplers/greedy.py
import torch
import torch.nn.functional as F

from inference.stop_utils import prepare_stop_sequences, check_stop_sequences
from inference.prompt_removal import remove_prompt_prefix

def greedy_generate_torch(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2000,
    stop_sequences=None,
    remove_prompt: bool = True
):
    """
    Token-by-token greedy decoding in Torch,
    with optional stop sequences ANYWHERE and optional prompt-prefix removal.
    """
    # Prepare stop sequences
    stop_seqs = prepare_stop_sequences(stop_sequences)

    eos_id = tokenizer.eos_token_id
    tokens = tokenizer.encode(prompt)
    if not tokens:
        if eos_id is None:
            raise ValueError("No tokens found and no eos_token_id in tokenizer.")
        tokens = [eos_id]

    device = getattr(model, "device", torch.device("cpu"))

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids).logits  # [1, seq_len, vocab_size]

        # Last step => [1, vocab_size]
        last_logits = logits[:, -1, :]
        next_token_id = torch.argmax(last_logits, dim=-1).item()
        tokens.append(next_token_id)

        # If EOS => break
        if eos_id is not None and next_token_id == eos_id:
            break

        # Check for stops
        current_text = tokenizer.decode(tokens)
        maybe_truncated = check_stop_sequences(current_text, stop_seqs)
        if maybe_truncated is not None:
            return maybe_truncated

    # Done => decode
    raw_output = tokenizer.decode(tokens)
    # remove prompt prefix if needed
    if remove_prompt:
        final_output = remove_prompt_prefix(raw_output, prompt)
        return final_output
    return raw_output
