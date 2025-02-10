import logging

from mlx_lm import generate
from mlx_lm.utils import make_sampler
from inference.chat_template import build_chat_prompt, remove_special_tokens_from_text

logger = logging.getLogger(__name__)

def mlx_chat_generation(
    loaded_model,
    loaded_tokenizer,
    system_prompt: str,
    user_messages: list,
    assistant_messages: list,
    max_new_tokens: int,
    sampler: str = "default",
    temperature: float = 0.6,
    top_p: float = 0.95,
    min_p: float = 0.0,
    min_tokens_to_keep: int = 1,
    stop_sequences=None,
    use_chat_template: bool = False,
    stream: bool = False,
):
    """
    1) Build the conversation prompt with build_chat_prompt(...)
    2) Convert stop_sequences -> extra EOS tokens (older MLX doesn't accept stop_sequences)
    3) Use make_sampler(...) with positional arguments to avoid older MLX keyword errors
    4) Pass that sampler to generate(...)
    5) Return cleaned text
    """

    # 1) Build the prompt
    final_prompt = build_chat_prompt(
        tokenizer=loaded_tokenizer,
        system_prompt=system_prompt,
        user_messages=user_messages,
        assistant_messages=assistant_messages,
        add_generation_prompt=True,
        use_template=use_chat_template,
    )
    logger.debug("mlx_chat_generation => final prompt:\n%r", final_prompt)

    # 2) If we have stop sequences, add them as extra EOS tokens
    if stop_sequences:
        for seq in stop_sequences:
            loaded_tokenizer.add_eos_token(seq)
        logger.debug("Added extra EOS tokens for stop sequences: %s", stop_sequences)

    # 3) Create the sampler object with positional params
    if sampler == "default":
        # "Greedy": temperature=0.0, top_p=1.0, but still pass min_p/min_tokens_to_keep
        logger.debug("Using GREEDY decode => make_sampler(0.0, 1.0, %.2f, %d).", min_p, min_tokens_to_keep)
        use_sampler = make_sampler(0.0, 1.0, min_p, min_tokens_to_keep)
    elif sampler == "top_p":
        # "Top-p": user-provided temperature, top_p, plus min_p, min_tokens_to_keep
        logger.debug("Using TOP-P => make_sampler(%.2f, %.2f, %.2f, %d).",
                     temperature, top_p, min_p, min_tokens_to_keep)
        use_sampler = make_sampler(temperature, top_p, min_p, min_tokens_to_keep)
    else:
        raise ValueError(f"Unsupported sampler: {sampler}")

    # 4) Prepare kwargs for generate. We do NOT pass 'stop_sequences', 'temperature', etc.
    generate_kwargs = {
        "model": loaded_model,
        "tokenizer": loaded_tokenizer,
        "prompt": final_prompt,
        "max_tokens": max_new_tokens,
        "sampler": use_sampler,   # We pass our sampler object here
        "verbose": bool(stream),  # If True, prints tokens as they are generated
    }

    # 5) Call MLX’s generate
    output_text = generate(**generate_kwargs)
    logger.debug("mlx_chat_generation => raw model output:\n%r", output_text)

    # 6) Remove any special placeholders
    cleaned_output = remove_special_tokens_from_text(output_text)
    logger.debug("mlx_chat_generation => cleaned output:\n%r", cleaned_output)

    return cleaned_output
