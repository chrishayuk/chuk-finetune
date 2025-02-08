# src/inference/mlx/mlx_chat_generation.py
import logging
import re
import mlx.core as mx

# imports
from inference.mlx.custom_generate_mlx import top_p_generate
from inference.chat_template import build_chat_prompt

# logger
logger = logging.getLogger(__name__)

def stream_greedy_generate(
    model,
    tokenizer,
    prompt: str,
    max_new_tokens: int = 256,
    stop_sequences=None
):
    if stop_sequences is None:
        stop_sequences = []

    tokens = tokenizer.encode(prompt)
    if not tokens:
        eos_id = tokenizer.eos_token_id
        tokens = [eos_id] if eos_id is not None else [0]

    for _ in range(max_new_tokens):
        logits = model(mx.array(tokens, mx.uint32)[None])
        last_logits = logits[:, -1, :]

        next_token = mx.argmax(last_logits, axis=-1).item()
        tokens.append(next_token)

        eos_id = tokenizer.eos_token_id
        if eos_id is not None and next_token == eos_id:
            break

        current_text = tokenizer.decode(tokens)
        for seq in stop_sequences:
            idx = current_text.find(seq)
            if idx != -1:
                current_text = current_text[:idx]
                return current_text

    return tokenizer.decode(tokens)

def remove_special_tokens_from_text(text: str, pattern=r"<\|.*?\|>") -> str:
    """
    Generic approach to remove <|...|> tokens from final text.
    """
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()

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
    stop_sequences=None,
    use_chat_template: bool = False
):
    """
    1) call build_chat_prompt(tokenizer, system_prompt, user_messages, assistant_messages, add_generation_prompt=True, use_template=use_chat_template)
    2) do streaming greedy or top_p
    3) remove <|...|> tokens
    4) return final
    """
    # 1) Build the conversation prompt
    final_prompt = build_chat_prompt(
        tokenizer=loaded_tokenizer,
        system_prompt=system_prompt,
        user_messages=user_messages,
        assistant_messages=assistant_messages,
        add_generation_prompt=True,
        use_template=use_chat_template
    )
    logger.debug("mlx_chat_generation => final prompt:\n%r", final_prompt)

    # 2) Sampler logic
    if sampler == "default":
        logger.debug("mlx_chat_generation => streaming GREEDY decode.")
        output_text = stream_greedy_generate(
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            prompt=final_prompt,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences
        )
    elif sampler == "top_p":
        logger.debug("mlx_chat_generation => top-p sampling.")
        output_text = top_p_generate(
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            prompt=final_prompt,
            max_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences
        )
    else:
        raise ValueError(f"Unsupported sampler: {sampler}")
    
    # debug
    logger.debug("mlx_chat_generation => raw model output:\n%r", output_text)

    # 3) Remove <|...|> tokens from final text
    cleaned_output = remove_special_tokens_from_text(output_text)
    logger.debug("mlx_chat_generation => cleaned output:\n%r", cleaned_output)

    # 4) Return
    return cleaned_output
