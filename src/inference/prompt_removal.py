# src/inference/prompt_removal.py
def remove_prompt_prefix(generated_text: str, prompt_text: str) -> str:
    """
    If 'generated_text' starts with 'prompt_text' exactly,
    remove that entire prefix from the output, then strip leading whitespace.
    """
    if generated_text.startswith(prompt_text):
        return generated_text[len(prompt_text):].lstrip()
    return generated_text
