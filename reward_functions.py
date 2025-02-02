import re
import requests
import difflib
from cli.train.logger_config import logger, CYAN, YELLOW
from src.verifiers.response_verifier import check_format, extract_answer
from mlx_lm import generate  # Assumes your generate function is imported

# Define a new color for evaluator responses (MAGENTA)
EVAL_COLOR = "\033[95m"
RESET = "\033[0m"

# Global evaluator model/tokenizer for self reward calculation.
EVAL_MODEL = None
EVAL_TOKENIZER = None

def set_eval_model(model, tokenizer):
    """
    Sets the global evaluator model and tokenizer.
    Call this during initialization so that the self reward function uses them.
    """
    global EVAL_MODEL, EVAL_TOKENIZER
    EVAL_MODEL = model
    EVAL_TOKENIZER = tokenizer
    logger.debug("Evaluation model and tokenizer have been set.")

def color_text(text, color):
    """
    Wraps text in ANSI color codes.
    """
    return f"{color}{text}{RESET}"

def local_format_calculate_reward(response_text: str, item: dict) -> float:
    """
    Returns a single float score for the presence of <think> and <answer>.
    """
    format_score = check_format(response_text)
    logger.debug(f"Local format reward: {format_score} for response: {response_text}")
    return format_score

def calculate_reward(response_text: str, item: dict):
    """
    Wraps local_format_calculate_reward to return a tuple: (score, feedback_text).
    """
    score = local_format_calculate_reward(response_text, item)
    
    expected_format = (
        "Expected format:\n"
        "<think> ... </think>\n"
        "<answer> ... </answer>"
    )
    
    errors = []
    if not re.search(r'<think>(.*?)</think>', response_text, re.DOTALL):
        errors.append("Missing <think> tag.")
    if not re.search(r'<answer>(.*?)</answer>', response_text, re.DOTALL):
        errors.append("Missing <answer> tag.")
    
    if errors:
        feedback_text = f"{expected_format}\nErrors: {' '.join(errors)}"
    else:
        feedback_text = f"{expected_format}\nAll required tags present."
    
    logger.debug(f"Local format feedback: {feedback_text}")
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
        score = data.get("score", 0.0)
        logger.debug(f"Remote reward score: {score} for response: {response_text}")
        return score
    except Exception as e:
        logger.warning(f"Remote verifier error: {e}")
        return 0.0

def sft_calculate_reward(response_text: str, item: dict) -> tuple:
    """
    Performs an SFT‐style check by comparing the full model response with the expected completion
    using a similarity measure. This version scales the raw similarity so that low raw values
    (e.g. 2%) can be boosted into a more useful reward range.
    
    Returns a tuple: (score, feedback_text).
    """
    expected_completion = item.get("completion", "").strip()
    actual_response = response_text.strip()
    
    if not expected_completion and not actual_response:
        return 1.0, "Both expected and actual responses are empty; considered a perfect match."
    
    raw_similarity = difflib.SequenceMatcher(None, expected_completion, actual_response).ratio()
    scale = 10.0  # Scale factor to boost low raw similarity values.
    scaled_score = min(1.0, raw_similarity * scale)
    feedback = f"Scaled similarity: {scaled_score*100:.1f}% (raw similarity: {raw_similarity*100:.1f}%)"
    logger.debug(f"SFT reward feedback: {feedback}")
    return scaled_score, feedback

def self_calculate_reward(response_text: str, item: dict) -> tuple:
    """
    Uses a separate evaluator model to calculate a reward by comparing the generated response
    with the expected completion. This function constructs an evaluation prompt that instructs the evaluator
    to consider completeness, conciseness, additional information, relevance, and consistency with the expected answer.
    
    The evaluator is instructed to output ONLY a single numeric score between 0 and 1.
    
    Returns a tuple: (score, feedback_text) that includes the raw evaluator response for debugging.
    """
    expected_completion = item.get("completion", "").strip()
    eval_prompt = (
        "You are a highly objective evaluator. Compare the following texts and evaluate how well the Model Response "
        "aligns with the Expected Answer based on these criteria:\n\n"
        "1. Completeness: Does the response include all essential information from the Expected Answer?\n"
        "2. Conciseness: Is the response appropriately brief (if the Expected Answer is brief) or sufficiently detailed (if the Expected Answer is long)?\n"
        "3. Additional Information: Does the response provide useful extra context without adding irrelevant details?\n"
        "4. Relevance: Does the response directly and accurately answer the prompt?\n"
        "5. Consistency: Is the response in line with the Expected Answer in tone, style, and content?\n\n"
        "After evaluating these factors, output ONLY a single numeric score between 0 and 1, where 1 indicates a perfect match "
        "and 0 indicates no match at all.\n\n"
        "Expected Answer: {}\n\n"
        "Model Response: {}\n\n"
        "Score:".format(expected_completion, response_text)
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
        # Extract the first numeric value from the output.
        num_match = re.search(r'([0-9]+(?:\.[0-9]+)?)', eval_response)
        if num_match:
            score = float(num_match.group(1))
        else:
            raise ValueError(f"No numeric score found in eval response: {eval_response}")
        # Color the evaluator response using the evaluator color.
        colored_eval_response = color_text(eval_response, EVAL_COLOR)
        feedback = f"Self reward: {score:.2f} | Evaluator response: {colored_eval_response}"
    except Exception as e:
        logger.warning(f"Self reward calculation failed: {e}")
        score = 0.0
        feedback = f"Self reward calculation failed: {e}"
    
    logger.debug(f"Self reward feedback: {feedback}")
    return score, feedback

def combined_calculate_reward(response_text: str, item: dict):
    """
    For items containing a "completion" key, use the self-reward mechanism (with the evaluator model)
    to assess the response. Otherwise, fall back to the local format check.
    
    Returns a tuple: (score, feedback_text).
    """
    if "completion" in item:
        return self_calculate_reward(response_text, item)
    else:
        local_score, local_feedback = calculate_reward(response_text, item)
        return local_score, local_feedback
