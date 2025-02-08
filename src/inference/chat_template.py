# src/inference/chat_template.py
import logging

# logging
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
    Build a consolidated chat prompt from optional system prompt, 
    multiple user messages, and assistant messages, 
    forming a multi-turn conversation.

    - If 'use_template=True' AND 'tokenizer.chat_template' and 'tokenizer.apply_chat_template'
      exist, we let the tokenizer build the prompt.
    - Otherwise, we do a fallback approach: 
         system_prompt\n\n
         User: user1
         Assistant: assist1
         User: user2
         ...
         Assistant:
      for the final user if there's an extra user message.

    :param tokenizer: The tokenizer (potentially with a built-in chat template).
    :param system_prompt: An optional system instruction for the AI.
    :param user_messages: A list of user messages (strings).
    :param assistant_messages: A list of assistant messages (strings).
    :param add_generation_prompt: Whether to append a final generation marker 
                                  when using the built-in chat template.
    :param use_template: If True, we try to call tokenizer.apply_chat_template. 
                         If it fails or is missing, fallback is used.
    :return: A single string ready to be tokenized and passed to the model.
    """
    # 1) Build a list of role-based messages
    messages = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    user_msgs = user_messages or []
    assistant_msgs = assistant_messages or []
    max_turns = max(len(user_msgs), len(assistant_msgs))
    for i in range(max_turns):
        if i < len(user_msgs):
            messages.append({"role": "user", "content": user_msgs[i]})
        if i < len(assistant_msgs):
            messages.append({"role": "assistant", "content": assistant_msgs[i]})

    # 2) If 'use_template' = True, try built-in chat template
    if use_template and hasattr(tokenizer, "chat_template") and tokenizer.chat_template and hasattr(tokenizer, "apply_chat_template"):
        logger.debug("build_chat_prompt => using tokenizer.apply_chat_template.")
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=add_generation_prompt
        )
        return text

    # 3) Otherwise fallback
    logger.debug("build_chat_prompt => fallback approach (User:\nAssistant:).")
    prompt = ""
    
    # system prompt
    if system_prompt:
        prompt += f"{system_prompt}\n\n"

    # interleave user & assistant
    # we already built messages, but let's just do a direct approach:
    fallback_user_msgs = []
    fallback_assist_msgs = []

    # Re-split the messages for fallback (since they might contain system as well)
    user_index = 0
    assist_index = 0

    # We'll just use user_msgs/assistant_msgs directly
    for u_msg, a_msg in zip(user_msgs, assistant_msgs):
        prompt += f"User: {u_msg}\nAssistant: {a_msg}\n"

    # If leftover user message:
    if len(user_msgs) > len(assistant_msgs):
        prompt += f"User: {user_msgs[-1]}\nAssistant:"

    # return the prompt
    return prompt
