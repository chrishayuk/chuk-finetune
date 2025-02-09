# src/verifiers/self_reward.py
import logging
import re

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

def self_calculate_reward(response_text: str, item: dict) -> tuple:
    """
    Uses an evaluator model to produce a reward score by comparing the generated response
    with the expected completion (when 'completion' is present).

    The evaluator is instructed to output ONLY a single numeric score between 0 and 1.

    Returns a tuple: (score, feedback_text).
    """
    # Example import for generation (adjust to your library)
    # from mlx_lm import generate
    # Or your own generation function if it's local

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

        # Replace `generate` with your actual inference call for EVAL_MODEL/EVAL_TOKENIZER
        # e.g.:
        # eval_response = generate(
        #     model=EVAL_MODEL,
        #     tokenizer=EVAL_TOKENIZER,
        #     prompt=eval_prompt,
        #     max_tokens=250,
        #     verbose=False
        # ).strip()

        # For demonstration, let's assume eval_response is returned from somewhere:
        eval_response = "0.75"  # Replace with the actual generation result
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
