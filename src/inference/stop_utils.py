# src/inference/stop_utils.py
import re

def make_stop_regex(stop_seq: str):
    """
    Convert a stop sequence (like 'User:' or '</answer>') into a regex
    that matches it anywhere, plus optional trailing whitespace.
    e.g. 'User:' => r'User:\s*'
    """
    # Escape the sequence, then allow trailing whitespace
    escaped_seq = re.escape(stop_seq) + r"\s*"
    pattern = re.compile(escaped_seq)
    return pattern

def prepare_stop_sequences(stop_sequences):
    """
    Compile each stop sequence into a regex that matches the sequence
    plus optional trailing whitespace, and return them in a list.
    """
    result = []
    for seq in (stop_sequences or []):
        pattern = make_stop_regex(seq)
        result.append((pattern, seq))
    return result

def check_stop_sequences(decoded_text: str, stop_sequences):
    """
    Check if 'decoded_text' contains any of the compiled stop-sequence regexes.
    If found, we truncate the text at the earliest match index and return it.
    Otherwise, return None.
    """
    earliest_idx = None
    found_seq = None
    for pattern, original_seq in stop_sequences:
        match = pattern.search(decoded_text)
        if match:
            idx = match.start()
            if earliest_idx is None or idx < earliest_idx:
                earliest_idx = idx
                found_seq = original_seq

    if earliest_idx is not None:
        truncated = decoded_text[:earliest_idx]
        # Debug print for clarity
        print(f"[DEBUG] Found stop seq '{found_seq}' at index {earliest_idx}. "
              f"Truncating => {repr(truncated[-60:])} (last 60 chars)")
        return truncated

    # Optional debug if no match is found
    # print("[DEBUG] No stop sequences found in =>", repr(decoded_text[-60:]))

    return None
