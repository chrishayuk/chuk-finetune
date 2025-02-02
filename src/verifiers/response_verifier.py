import re

def check_format(response: str) -> float:
    """
    Returns a score based on which tags are present:
      - <think> -> 0.5
      - <answer> -> 0.5
    If neither tag is present, we apply a small penalty of -0.1.
    """
    has_think = bool(re.search(r'<think>(.*?)</think>', response, re.DOTALL))
    has_answer = bool(re.search(r'<answer>(.*?)</answer>', response, re.DOTALL))

    score = 0.0
    if has_think:
        score += 0.5
    if has_answer:
        score += 0.5

    # If neither tag is present, assign negative penalty
    if score == 0.0:
        score = -0.1

    return score

def extract_answer(response: str) -> str:
    """
    Extracts the text inside <answer>...</answer>.
    Returns None if the tag is not present.
    """
    match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
    return match.group(1).strip() if match else None

def calculate_reward(response: str, verifier) -> float:
    """
    1) Awards partial format reward for <think> and <answer>.
    2) Also checks correctness of the <answer> using verifier.check().
    3) Returns the sum of the format score and the accuracy score.
    """
    # (1) Check format
    format_score = check_format(response)

    # (2) Extract the <answer> for correctness check
    model_answer = extract_answer(response)

    # (3) Accuracy reward (assumes verifier.check() returns True/False)
    if model_answer and verifier.check(model_answer):
        accuracy_score = 1.0
    else:
        accuracy_score = 0.0

    return format_score + accuracy_score
