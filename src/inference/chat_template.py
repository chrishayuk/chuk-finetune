# src/inference/chat_template.py
def build_chat_prompt(
    tokeniser,
    system_prompt: str = None,
    user_messages: list = None,
    assistant_messages: list = None,
    add_generation_prompt: bool = True
) -> str:
    """
    Build a consolidated chat prompt from an optional system message, multiple user messages,
    and multiple assistant messages. The function interleaves user and assistant messages 
    in the order they appear, producing a multi-turn conversation.

    Example:
        system_prompt = "You are a helpful and knowledgeable assistant."
        user_messages = [
            "Hello, how are you?",
            "What is the capital of France?"
        ]
        assistant_messages = [
            "Hi there! How can I help you today?"
        ]

    This will create a conversation:
      - system: "You are a helpful and knowledgeable assistant."
      - user: "Hello, how are you?"
      - assistant: "Hi there! How can I help you today?"
      - user: "What is the capital of France?"
    
    :param tokeniser: The tokeniser with a built-in apply_chat_template method.
    :param system_prompt: An optional system instruction for the AI.
    :param user_messages: A list of user messages (strings).
    :param assistant_messages: A list of assistant messages (strings).
    :param add_generation_prompt: Whether to append a final generation marker to prompt.
    :return: A single string ready to be tokenised and passed to the model.
    """
    messages = []

    # If there is a system prompt, place it first
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})

    # Use empty lists if none are provided
    user_msgs = user_messages or []
    assistant_msgs = assistant_messages or []

    # Interleave user and assistant messages turn by turn
    max_turns = max(len(user_msgs), len(assistant_msgs))
    for i in range(max_turns):
        if i < len(user_msgs):
            messages.append({"role": "user", "content": user_msgs[i]})
        if i < len(assistant_msgs):
            messages.append({"role": "assistant", "content": assistant_msgs[i]})

    # Let the tokeniser handle final prompt formatting
    text = tokeniser.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=add_generation_prompt
    )

    # return the text
    return text
