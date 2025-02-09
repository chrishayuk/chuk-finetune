# src/inference/torch/custom_generate_torch.py
import torch
import torch.nn.functional as F

# import stop and prompt utils
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
    Token-by-token Top-p (nucleus) sampling in Torch, with optional stop sequences
    ANYWHERE in the text, plus optional removal of the prompt prefix at the end.
    """
    # 1) Prepare the stop-sequence regexes
    stop_seqs = prepare_stop_sequences(stop_sequences)

    # 2) Encode the prompt
    eos_id = tokenizer.eos_token_id
    tokens = tokenizer.encode(prompt)
    if not tokens:
        if eos_id is None:
            raise ValueError("No tokens found and no eos_token_id in tokenizer.")
        tokens = [eos_id]

    device = getattr(model, "device", torch.device("cpu"))

    # 3) Iteratively generate new tokens
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)

        with torch.no_grad():
            logits = model(input_ids).logits  # [1, seq_len, vocab_size]

        # Next-token logits => [1, vocab_size]
        next_token_logits = logits[:, -1, :]

        # Temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Convert to probabilities
        probs = F.softmax(next_token_logits, dim=-1)[0]  # shape: [vocab_size]

        # Sort ascending
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=False)

        # Cumulative sum
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Identify cutoff (top-p)
        cutoff_mask = cumulative_probs > (1.0 - top_p)
        truncated_probs = torch.where(cutoff_mask, sorted_probs, torch.tensor(0.0, device=probs.device))

        # Log-probs for sampling
        truncated_logprobs = torch.log(truncated_probs + 1e-12)

        # Sample from truncated distribution
        sample_idx = torch.multinomial(torch.exp(truncated_logprobs), 1).item()
        next_token_id = sorted_indices[sample_idx].item()

        tokens.append(next_token_id)

        # Check for EOS
        if eos_id is not None and next_token_id == eos_id:
            break

        # 4) Check stop sequences ANYWHERE
        current_text = tokenizer.decode(tokens)
        maybe_truncated = check_stop_sequences(current_text, stop_seqs)
        if maybe_truncated is not None:
            # Return truncated
            return maybe_truncated

    # 5) If we finish naturally => decode
    raw_output = tokenizer.decode(tokens)

    # 6) Optionally remove the prompt prefix if the model repeated it at the start
    if remove_prompt:
        final_output = remove_prompt_prefix(raw_output, prompt)
        return final_output
    else:
        return raw_output


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
    # 1) Prepare stop sequences
    stop_seqs = prepare_stop_sequences(stop_sequences)

    eos_id = tokenizer.eos_token_id
    tokens = tokenizer.encode(prompt)
    if not tokens:
        if eos_id is None:
            raise ValueError("No tokens found and no eos_token_id in tokenizer.")
        tokens = [eos_id]

    device = getattr(model, "device", torch.device("cpu"))

    # 2) Loop up to max_new_tokens
    for _ in range(max_new_tokens):
        input_ids = torch.tensor([tokens], dtype=torch.long, device=device)
        with torch.no_grad():
            logits = model(input_ids).logits  # [1, seq_len, vocab_size]

        # [1, vocab_size]
        last_logits = logits[:, -1, :]
        next_token_id = torch.argmax(last_logits, dim=-1).item()
        tokens.append(next_token_id)

        # EOS?
        if eos_id is not None and next_token_id == eos_id:
            break

        # 3) Check stop sequences ANYWHERE
        current_text = tokenizer.decode(tokens)
        maybe_truncated = check_stop_sequences(current_text, stop_seqs)
        if maybe_truncated is not None:
            return maybe_truncated

    # 4) Done => decode
    raw_output = tokenizer.decode(tokens)

    # 5) remove prompt prefix if desired
    if remove_prompt:
        final_output = remove_prompt_prefix(raw_output, prompt)
        return final_output
    else:
        return raw_output
