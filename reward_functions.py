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
    If there's an error connecting, returns None to indicate a failed attempt.
    """
    # Default URL can be overridden by per-verifier data
    url = item.get("verifier_url", "http://0.0.0.0:8000") + "/verify"
    payload = {
        "text": response_text,
        "verifier": item.get("verifier", "default"),
        "feedback": True
    }
    try:
        resp = requests.post(url, json=payload, timeout=10)
        resp.raise_for_status()
        data = resp.json()
        score = data.get("score", 0.0)
        logger.debug(f"Remote reward score: {score} for response: {response_text}")
        return score
    except Exception as e:
        logger.warning(f"Remote verifier error: {e}")
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

    # 2) If there are one or more verifiers, query each remote endpoint, collecting scores.
    if "verifiers" in item and item["verifiers"]:
        remote_scores = []
        for verifier_info in item["verifiers"]:
            # Make a shallow copy of item for the current verifier
            item_copy = dict(item)
            item_copy["verifier"] = verifier_info["name"]
            item_copy["verifier_url"] = verifier_info.get("url", "http://0.0.0.0:8000")

            score = remote_calculate_reward(response_text, item_copy)
            if score is not None:
                remote_scores.append(score)

        # If ALL verifiers failed (no valid scores), skip training for this item
        if not remote_scores:
            msg = "All remote verifiers failed (connection refused or other error). Skipping item."
            logger.warning(msg)
            return None, msg

        # Otherwise, average the results from all successful verifiers
        final_score = sum(remote_scores) / len(remote_scores)
        feedback = f"Averaged {len(remote_scores)} remote verifier score(s): {final_score:.2f}"
        return final_score, feedback

    # 3) No 'completion' or 'verifiers' -> skip entirely.
    return None, "No completion or verifiers provided; skipping."
