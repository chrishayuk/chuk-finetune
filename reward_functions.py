import re
import requests
import difflib
from cli.train.logger_config import logger
from mlx_lm import generate  # Your model generation function

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
        logger.warning("No explicit 'Assistant:' found; using entire response.")

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

    # 4) For the 'verifier_answer' verifier, extract the content within <verifier_answer>...</verifier_answer>
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


def self_calculate_reward(response_text: str, item: dict) -> tuple:
    """
    Uses an evaluator model to produce a reward score by comparing the generated response
    with the expected completion (when 'completion' is present).

    The evaluator is instructed to output ONLY a single numeric score between 0 and 1.

    Returns a tuple: (score, feedback_text).
    """
    expected_completion = item.get("completion", "").strip()
    eval_prompt = (
        "You are a highly objective evaluator. Compare the following texts and evaluate how well "
        "the Model Response aligns with the Expected Answer, considering completeness, conciseness, "
        "additional information, relevance, and consistency.\n\n"
        "After evaluating, output ONLY a single numeric score between 0 and 1.\n\n"
        f"Expected Answer: {expected_completion}\n\n"
        f"Model Response: {response_text}\n\n"
        "Score:"
    )
    logger.debug(f"Self reward evaluation prompt: {eval_prompt}")
    
    try:
        if EVAL_MODEL is None or EVAL_TOKENIZER is None:
            raise ValueError("Evaluation model or tokenizer not set.")
        eval_response = generate(
            model=EVAL_MODEL,
            tokenizer=EVAL_TOKENIZER,
            prompt=eval_prompt,
            max_tokens=250,
            verbose=False
        ).strip()
        logger.debug(f"Raw evaluator response: {eval_response}")

        num_match = re.search(r'([0-9]+(?:\.[0-9]+)?)', eval_response)
        if num_match:
            score = float(num_match.group(1))
        else:
            raise ValueError(f"No numeric score found in evaluator response: {eval_response}")

        coloured_eval_response = color_text(eval_response, EVAL_COLOR)
        feedback = f"Self reward: {score:.2f} | Evaluator response: {coloured_eval_response}"
    except Exception as e:
        logger.warning(f"Self reward calculation failed: {e}")
        score = 0.0
        feedback = f"Self reward calculation failed: {e}"
    
    logger.debug(f"Self reward feedback: {feedback}")
    return score, feedback


def combined_calculate_reward(response_text: str, item: dict):
    """
    Determines which reward calculation to perform:
      1) If 'completion' is present, run self_calculate_reward.
      2) Else if 'verifiers' is present, we call remote_calculate_reward for each,
         ignoring any that fail. If ALL fail, we return (None, "..."), indicating skip.
      3) Otherwise, return (None, "No completion or verifiers provided"), also indicating skip.
    
    Returns a tuple: (score, feedback_text), 
    but score can be None if we need to skip training for this item.
    """

    # 1) If there's a 'completion', use the evaluator-based self reward.
    if "completion" in item:
        return self_calculate_reward(response_text, item)

    # 2) If there are verifiers, query each remote endpoint, collecting scores.
    if "verifiers" in item and item["verifiers"]:
        remote_scores = []
        for verifier_info in item["verifiers"]:
            # Make a shallow copy of item for this verifier
            item_copy = dict(item)
            item_copy["verifier"] = verifier_info["name"]
            item_copy["verifier_url"] = verifier_info.get("url", "http://0.0.0.0:8000")

            # Important: Copy any verifier-specific args
            if "args" in verifier_info:
                item_copy["args"] = verifier_info["args"]

            score = remote_calculate_reward(response_text, item_copy)
            if score is not None:
                remote_scores.append(score)

        # If ALL verifiers failed
        if not remote_scores:
            msg = "All remote verifiers failed (connection refused or other error). Skipping item."
            logger.warning(msg)
            return None, msg

        # Otherwise, average the results
        final_score = sum(remote_scores) / len(remote_scores)
        feedback = f"Averaged {len(remote_scores)} remote verifier score(s): {final_score:.2f}"
        return final_score, feedback

    # 3) If no completion or verifiers, skip
    return None, "No completion or verifiers provided; skipping."
