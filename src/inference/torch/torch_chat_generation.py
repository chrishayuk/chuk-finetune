import logging

# imports
from inference.torch.custom_generate_torch import (greedy_generate_torch, top_p_generate_torch, top_p_generate_torch_with_kvcache)
from inference.chat_template import build_chat_prompt, remove_special_tokens_from_text

# logger
logger = logging.getLogger(__name__)


def torch_chat_generation(
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
    use_chat_template: bool = False,
    stream: bool = False
):
    """
    Main entry point:
      - If stream=True => partial tokens from 'TextIteratorStreamer' in real time.
      - If stream=False => single final string from manual generation code (greedy or top-p).
      - We pass attention_mask to generate() to avoid the "mask not set" warning.
    """
    # 1) Build final prompt
    final_prompt = build_chat_prompt(
        tokenizer=loaded_tokenizer,
        system_prompt=system_prompt,
        user_messages=user_messages,
        assistant_messages=assistant_messages,
        add_generation_prompt=True,
        use_template=use_chat_template
    )

    # debug
    logger.debug(f"Torch final prompt:\n{final_prompt}")

    # For your manual code, let's also pass attention_mask if you want
    # but your custom code might not use it. We'll just do the old approach:
    if sampler == "default":
        # greedy
        raw_output = greedy_generate_torch(
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            prompt=final_prompt,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_sequences
        )
    elif sampler == "top_p":
        # top p
        raw_output = top_p_generate_torch_with_kvcache(
            model=loaded_model,
            tokenizer=loaded_tokenizer,
            prompt=final_prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            stop_sequences=stop_sequences
        )
    else:
        raise ValueError(f"Unsupported sampler: {sampler}")
    
    # clean outputs
    logger.debug(f"Torch raw output:\n{raw_output}")
    cleaned_output = remove_special_tokens_from_text(raw_output)
    logger.debug(f"Torch cleaned output:\n{cleaned_output}")

    # return cleaned
    return cleaned_output
