# src/inference/stop_utils.py
import re

def make_stop_regex(stop_seq: str):
    """
    Given a stop sequence like 'User:', return a regex that matches it anywhere.
    Example: 'User:' => re.compile(r'User:')
    """
    # escape sequence match
    escaped_seq = re.escape(stop_seq)

    # return the match
    return re.compile(escaped_seq)

def prepare_stop_sequences(stop_sequences):
    """
    Turn each string in 'stop_sequences' into a compiled regex that
    matches anywhere in the text.
    Returns a list of (compiled_regex, original_seq).
    """
    result = []
    for seq in (stop_sequences or []):
        pattern = make_stop_regex(seq)
        result.append((pattern, seq))
    return result

def check_stop_sequences(decoded_text: str, stop_sequences):
    """
    Check if 'decoded_text' contains any of the stop-sequence regexes.
    If found, we truncate the text at the earliest match index.
    Return the truncated text if a match is found, otherwise None.
    """
    earliest_idx = None
    for pattern, original_seq in stop_sequences:
        match = pattern.search(decoded_text)
        if match:
            idx = match.start()
            if earliest_idx is None or idx < earliest_idx:
                earliest_idx = idx

    if earliest_idx is not None:
        return decoded_text[:earliest_idx]

    return None
