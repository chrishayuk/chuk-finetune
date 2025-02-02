import re
import requests
from logger_config import logger, color_text, CYAN, YELLOW
from src.verifiers.response_verifier import check_format, extract_answer

def local_format_calculate_reward(response_text: str, item: dict) -> float:
    """
    Returns a single float score for the presence of <think> and <answer>.
    
    For example, check_format might add:
      - 0.5 if <think> is present
      - 0.5 if <answer> is present
    and return -0.1 if neither is found.
    """
    format_score = check_format(response_text)
    return format_score

def calculate_reward(response_text: str, item: dict):
    """
    Wraps local_format_calculate_reward to return a tuple:
      (score, feedback_text)
    
    The feedback text includes the expected format and explicitly lists errors
    (missing tags) if the response does not follow the expected format.
    """
    score = local_format_calculate_reward(response_text, item)
    
    # Define the expected format
    expected_format = (
        "Expected format:\n"
        "<think> ... </think>\n"
        "<answer> ... </answer>"
    )
    
    # Check for errors using regular expressions
    errors = []
    if not re.search(r'<think>(.*?)</think>', response_text, re.DOTALL):
        errors.append("Missing <think> tag.")
    if not re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL):
        errors.append("Missing <answer> tag.")
    
    # Construct the feedback string based on errors
    if errors:
        feedback_text = f"{expected_format}\nErrors: {' '.join(errors)}"
    else:
        feedback_text = f"{expected_format}\nAll required tags present."
    
    return score, feedback_text

def remote_calculate_reward(response_text: str, item: dict) -> float:
    """
    Queries a remote verifier endpoint for a float score.
    If there's an error, returns 0.0.
    """
    url = item.get("verifier_url", "http://0.0.0.0:8000") + "/verify"
    payload = {
        "text": response_text,
        "verifier": item["verifier"],
        "feedback": True
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        return data.get("score", 0.0)
    except Exception as e:
        logger.warning(f"Remote verifier error: {e}")
        return 0.0

def combined_calculate_reward(response_text: str, item: dict):
    """
    Combines local format and remote scores if desired.
    Currently, it returns the local score and feedback.
    
    To enable remote scoring, uncomment and combine the values.
    """
    local_score, local_feedback = calculate_reward(response_text, item)
    # remote_score = remote_calculate_reward(response_text, item)
    # total_score = local_score + remote_score
    # combined_feedback = f"Local: {local_feedback} | Remote: {remote_feedback}"
    # return total_score, combined_feedback

    return local_score, local_feedback
