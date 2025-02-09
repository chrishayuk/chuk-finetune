# src/verifiers/combined_reward.py
import logging

# imports
from verifiers.remote_reward import remote_calculate_reward
from verifiers.self_reward import self_calculate_reward

# set the logger
logger = logging.getLogger(__name__)

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
