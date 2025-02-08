# src/inference/torch/custom_generate_torch.py
import torch
import torch.nn.functional as F

def top_p_generate_torch(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2000,
    temperature: float = 0.6,
    top_p: float = 0.95
):
    """
    Token-by-token Top-p (nucleus) sampling in Torch, adapted from `greedy_generate_torch`.
    By default, uses temperature=0.6 and top_p=0.95.

    :param model: A Hugging Face (or compatible) Torch model returning logits of shape [1, seq_len, vocab_size].
    :param tokenizer: The tokenizer for encoding/decoding text.
    :param prompt: The text prompt to start generation from.
    :param max_new_tokens: The maximum number of tokens to generate beyond the prompt.
    :param temperature: Sampling temperature (>= 0.0). Larger => more random.
    :param top_p: Probability threshold for nucleus sampling. 
                  0.95 means we sample from the smallest set of tokens whose cumulative probability >= 0.05.
    :return: Decoded string of tokens (prompt + newly generated content).
    """

    # 1) Encode the prompt
    eos_id = tokenizer.eos_token_id
    tokens = tokenizer.encode(prompt)
    if not tokens:
        if eos_id is None:
            raise ValueError("No tokens found and no eos_token_id in tokenizer.")
        tokens = [eos_id]

    # 2) Iteratively generate new tokens
    for _ in range(max_new_tokens):
        # shape [1, seq_len]
        input_ids = torch.tensor([tokens], dtype=torch.long, device=model.device)

        with torch.no_grad():
            # shape => [1, seq_len, vocab_size]
            logits = model(input_ids).logits

        # Last step logits => shape [1, vocab_size]
        next_token_logits = logits[:, -1, :]

        # Apply temperature
        if temperature != 1.0:
            next_token_logits = next_token_logits / temperature

        # Convert logits -> probabilities
        probs = F.softmax(next_token_logits, dim=-1)  # shape [1, vocab_size]

        # We assume batch size = 1, so let's flatten it for clarity
        probs = probs[0]                # shape: [vocab_size]

        # Sort tokens by ascending probability
        sorted_probs, sorted_indices = torch.sort(probs, dim=-1, descending=False)
        # sorted_probs, sorted_indices both shape: [vocab_size], ascending order

        # Cumulative sum of sorted probabilities
        cumulative_probs = torch.cumsum(sorted_probs, dim=-1)

        # Identify cutoff where sum >= (1 - top_p)
        cutoff_mask = cumulative_probs > (1.0 - top_p)

        # Zero out everything below that threshold
        truncated_probs = torch.where(cutoff_mask, sorted_probs, torch.tensor(0.0, device=probs.device))

        # Convert to log-probs for sampling
        truncated_logprobs = torch.log(truncated_probs + 1e-12)

        # Sample from the truncated distribution
        sampled_idx = torch.multinomial(torch.exp(truncated_logprobs), 1).item()

        # Map from sorted index back to real token ID
        next_token_id = sorted_indices[sampled_idx].item()

        # Append to sequence
        tokens.append(next_token_id)

        # If we hit EOS => break
        if eos_id is not None and next_token_id == eos_id:
            break

    # 3) Decode the entire sequence
    return tokenizer.decode(tokens)


#
# (Optional) If you still want a separate greedy function, you can keep it:
#

def greedy_generate_torch(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2000
):
    """
    Original token-by-token greedy decoding in Torch.
    """
    # (Same code you had before, unchanged)
    eos_id = tokenizer.eos_token_id
    tokens = tokenizer.encode(prompt)
    if not tokens:
        if eos_id is None:
            raise ValueError("No tokens found and no eos_token_id in tokenizer.")
        tokens = [eos_id]

    for _ in range(max_new_tokens):
        input_ids = torch.tensor([tokens], dtype=torch.long, device=model.device)
        with torch.no_grad():
            logits = model(input_ids).logits
        last_logits = logits[:, -1, :]
        next_token = torch.argmax(last_logits, dim=-1).item()
        tokens.append(next_token)
        if eos_id is not None and next_token == eos_id:
            break

    return tokenizer.decode(tokens)
