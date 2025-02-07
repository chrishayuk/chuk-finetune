# src/inference/torch/custom_generate_torch.py
import torch

def greedy_generate_torch(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 2000
):
    """
    Token-by-token greedy decoding in Torch, mimicking MLX's `greedy_generate`.

    :param model: A Hugging Face (or compatible) Torch model returning logits of shape [1, seq_len, vocab_size].
    :param tokenizer: The tokenizer for encoding/decoding text.
    :param prompt: The text prompt to start generation from.
    :param max_tokens: The maximum number of tokens to generate beyond the prompt.
    :return: Decoded string of tokens (prompt + generated content).
    """

    # 1) Encode the prompt
    tokens = tokenizer.encode(prompt)
    # If empty => fallback to eos (like MLX does)
    eos_id = tokenizer.eos_token_id
    if not tokens:
        if eos_id is None:
            raise ValueError("No tokens found and no eos_token_id in tokenizer.")
        tokens = [eos_id]

    # 2) Iteratively generate
    for _ in range(max_new_tokens):
        # shape [1, seq_len]
        input_ids = torch.tensor([tokens], dtype=torch.long, device=model.device)

        with torch.no_grad():
            # shape => [1, seq_len, vocab_size]
            logits = model(input_ids).logits

        # Last logits => shape [1, vocab_size]
        last_logits = logits[:, -1, :]

        # Argmax for greedy
        next_token = torch.argmax(last_logits, dim=-1).item()

        tokens.append(next_token)

        # If we see EOS => break
        if eos_id is not None and next_token == eos_id:
            break

    # 3) Decode the entire sequence
    return tokenizer.decode(tokens)
