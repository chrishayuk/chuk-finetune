# src/inference/chat_template.py

import logging
import re

logger = logging.getLogger(__name__)

def build_chat_prompt(
    tokenizer,
    system_prompt: str = None,
    user_messages: list = None,
    assistant_messages: list = None,
    add_generation_prompt: bool = True,
    use_template: bool = True
) -> str:
    """
    Build a consolidated chat prompt OR a single-turn prompt,
    depending on 'use_template'.

    1) If 'use_template=False', we skip all chat formatting:
         - Just return system_prompt + the last user message (or all user messages).
    2) If 'use_template=True' AND the tokenizer has a built-in template
       (tokenizer.chat_template + tokenizer.apply_chat_template),
       we use that.
    3) Otherwise, fallback to a manual "User:\nAssistant:" style.
    """
    user_messages = user_messages or []
    assistant_messages = assistant_messages or []

    ################################################################
    # 1) Non-chat (single-turn or plain) if use_template=False
    ################################################################
    if not use_template:
        logger.debug("build_chat_prompt => 'use_template=False': non-chat prompt.")
        prompt = ""

        # Optional system prompt
        if system_prompt:
            prompt += f"{system_prompt.strip()}\n\n"

        # Single-turn (or just plain) approach:
        # You can choose if you want only the last user message, or all combined.
        if user_messages:
            # Example: Just append the last user message
            prompt += user_messages[-1].strip()

        # If you want a marker or an extra newline at the end:
        if add_generation_prompt:
            prompt += "\n"

        return prompt

    ################################################################
    # 2) Chat mode with possible built-in template
    ################################################################
    # Build the role-based list of messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    max_turns = max(len(user_messages), len(assistant_messages))
    for i in range(max_turns):
        if i < len(user_messages):
            messages.append({"role": "user", "content": user_messages[i]})
        if i < len(assistant_messages):
            messages.append({"role": "assistant", "content": assistant_messages[i]})

    # If the tokenizer has a built-in chat template and we want to use it
    if (
        hasattr(tokenizer, "chat_template") and tokenizer.chat_template
        and hasattr(tokenizer, "apply_chat_template")
    ):
        logger.debug("build_chat_prompt => using tokenizer.apply_chat_template.")
        return tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )

    ################################################################
    # 3) Fallback: manual "User:\nAssistant:" style
    ################################################################
    logger.debug("build_chat_prompt => fallback approach (User:\nAssistant:).")
    prompt = ""

    # System prompt first
    if system_prompt:
        prompt += f"{system_prompt}\n\n"

    # Interleave user & assistant
    for u_msg, a_msg in zip(user_messages, assistant_messages):
        prompt += f"User: {u_msg}\nAssistant: {a_msg}\n"

    # If there's one extra user message at the end
    if len(user_messages) > len(assistant_messages):
        prompt += f"User: {user_messages[-1]}\nAssistant:"

    return prompt


def remove_special_tokens_from_text(text: str, pattern=r"<\|.*?\|>") -> str:
    """
    Generic approach to remove <|...|> tokens from final text.
    """
    return re.sub(pattern, "", text, flags=re.DOTALL).strip()
