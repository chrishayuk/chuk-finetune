import logging
import re
import requests

# set the logger
logger = logging.getLogger(__name__)

# ANSI colour for evaluator responses
EVAL_COLOR = "\033[95m"
RESET = "\033[0m"

# Global evaluator model/tokenizer for self reward calculation.
EVAL_MODEL = None
EVAL_TOKENIZER = None

def set_eval_model(model, tokenizer):
    """
    Sets the global evaluator model and tokenizer.
    Call this during initialisation so that the self reward function uses them.
    """
    global EVAL_MODEL, EVAL_TOKENIZER
    EVAL_MODEL = model
    EVAL_TOKENIZER = tokenizer
    logger.debug("Evaluation model and tokenizer have been set.")


def color_text(text, color):
    """Wraps text in ANSI colour codes."""
    return f"{color}{text}{RESET}"

def remote_calculate_reward(response_text: str, item: dict) -> float or None:
    """
    Queries a remote verifier endpoint for a float score.
    Returns None on any connection/error issue to indicate a failed attempt.
    """

    # 1) Remove any trailing end-of-text markers and normalise spaces
    cleaned_text = response_text.replace("<|endoftext|>", "")
    cleaned_text = re.sub(r"\s+", " ", cleaned_text).strip()

    # 2) Extract only Assistant's portion (content after "Assistant:")
    assistant_pattern = r"Assistant:\s*(.*)"
    match = re.search(assistant_pattern, cleaned_text, re.DOTALL | re.IGNORECASE)
    if match:
        cleaned_text = match.group(1).strip()
        logger.debug(f"Extracted Assistant response: {cleaned_text}")
    else:
        logger.debug("No explicit 'Assistant:' found; using entire response.")

    # 3) For the format verifier, check the strict <think>...</think><answer>...</answer> structure
    if item.get("verifier") == "reasoning_format":
        format_pattern = r"<think>.*?</think>\s*<answer>.*?</answer>"
        if not re.search(format_pattern, cleaned_text, re.DOTALL):
            logger.warning("Format verifier: Missing or malformed <think>...</think><answer>...</answer> structure.")

    # 4) For the 'verifier_answer' verifier, extract the content within <verifier_answer>...</verifier_answer>
    if item.get("verifier") == "verifier_answer":
        pattern = r"<verifier_answer>\s*(.*?)\s*</verifier_answer>"
        matches = re.findall(pattern, cleaned_text, flags=re.DOTALL | re.IGNORECASE)
        extracted_text = None

        # Grab the last non-placeholder match
        for candidate in reversed(matches):
            candidate_clean = candidate.strip()
            if candidate_clean.lower() == "verifier answer here":
                continue
            if candidate_clean:
                extracted_text = candidate_clean
                break

        if extracted_text:
            logger.debug(f"Extracted <verifier_answer> text: {extracted_text}")
            cleaned_text = f"<verifier_answer>{extracted_text}</verifier_answer>"
        else:
            logger.warning("No valid <verifier_answer> content found. Using entire response for 'verifier_answer'.")

    # 4b) For the 'answer_satisfaction' verifier, extract the content within <answer>...</answer>
    if item.get("verifier") == "answer_satisfaction":
        pattern = r"<answer>\s*(.*?)\s*</answer>"
        matches = re.findall(pattern, cleaned_text, flags=re.DOTALL | re.IGNORECASE)
        extracted_text = None

        # Grab the last non-placeholder match
        for candidate in reversed(matches):
            candidate_clean = candidate.strip()
            if candidate_clean.lower() == "answer here":
                continue
            if candidate_clean:
                extracted_text = candidate_clean
                break

        if extracted_text:
            logger.debug(f"Extracted <answer> text: {extracted_text}")
            cleaned_text = f"<answer>{extracted_text}</answer>"
        else:
            logger.warning("No valid <answer> content found. Using entire response for 'answer'.")

    # 5) Prepare the payload for the remote verifier
    url = item.get("verifier_url", "http://0.0.0.0:8000") + "/verify"
    payload = {
        "text": cleaned_text,
        "verifier": item.get("verifier", "default"),
        "feedback": True
    }

    # If the item has args (e.g. gold_solution), include them
    if "args" in item and isinstance(item["args"], dict):
        payload["args"] = item["args"]
        logger.debug(f"Including args in payload: {item['args']}")

    logger.debug(f"Final text sent to verifier '{payload['verifier']}': {cleaned_text}")
    logger.debug(f"Sending to URL: {url}, payload: {payload}")

    # 6) Send the request and handle any exceptions
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        logger.debug(f"Verifier response data: {data}")

        score = data.get("score", 0.0)
        feedback = data.get("feedback", [])
        logger.info(f"Verifier score: {score}, feedback: {feedback}")
        return score
    except requests.RequestException as e:
        logger.warning(f"Error sending request to remote verifier: {e}")
    except Exception as e:
        logger.warning(f"Unexpected error in verifier call: {e}")

    # If we reach here, something went wrong, so return None
    return None
