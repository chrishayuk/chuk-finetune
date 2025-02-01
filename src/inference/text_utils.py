# src/inference/text_utils.py
import re

# constants
ROLE_REGEX = re.compile(r'^(assistant|user|system)\s*:?[\r\n]*', re.IGNORECASE)
MULTILINE_ROLE_REGEX = re.compile(r'^(assistant|user|system)\s*:?[\r\n]+', re.MULTILINE | re.IGNORECASE)

def strip_role_prefixes(line: str) -> str:
    """
    Strips a leading role label from a line.
    """
    return ROLE_REGEX.sub('', line)

def skip_repeated_line(line: str, known_lines: set) -> bool:
    """
    Returns True if the stripped line exactly appears in known_lines.
    """
    return line.strip() in known_lines

def final_cleanup(full_text: str, known_lines: set) -> str:
    """
    Removes leftover role labels and any repeated conversation lines from the final text.
    """
    text_no_roles = MULTILINE_ROLE_REGEX.sub('', full_text)
    final_lines = [
        line for line in text_no_roles.splitlines()
        if line.strip() not in known_lines
    ]
    return "\n".join(final_lines).strip()
