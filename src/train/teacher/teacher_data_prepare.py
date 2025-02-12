# src/train/teacher/teacher_data_prepare.py

import logging
from typing import Any, List, Dict, Optional, Tuple, Callable

logger = logging.getLogger(__name__)

def prepare_batch_data_for_teacher(
    batch_questions: List[Any],
    generate_single_fn: Callable[[str, bool], Tuple[str, float]],
    calculate_reward: Callable[[str, Dict[str, Any]], Tuple[Optional[float], str]],
    G: int = 4,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    A teacher-specific data-preparation function that:
      1) Loops over a batch of raw items,
      2) For each item, checks if it's a dict with a 'prompt' key; otherwise skips,
      3) Calls 'generate_single_fn(...)' G times per valid item, collecting teacher responses & log-probs,
      4) Runs 'calculate_reward' to optionally skip items if reward=None,
      5) Returns a list of data dicts, each containing:
         {
           "item": the original item dict,
           "responses": [...],
           "teacher_logprobs": [...],
           "rewards": [...],
         }

    This is analogous to 'grpo_prepare.py' but specialized for teacher data collection.

    Args:
        batch_questions: A list of raw items (dicts or possibly other structures).
        generate_single_fn: A function (prompt, verbose) => (response_text, teacher_logprob),
                            used to generate teacher responses.
        calculate_reward: A function (resp_text, item_dict) => (score, feedback);
                          returns None => skip item.
        G (int): Number of responses to generate per item.
        verbose (bool): Whether to print/log debug info.

    Returns:
        A list of dicts with teacher responses, log-probs, and optional rewards.
    """
    batch_data = []

    # Loop over each question/item in the batch
    for i, raw_item in enumerate(batch_questions):

        # Inline check: must be a dict containing 'prompt'
        if not isinstance(raw_item, dict) or "prompt" not in raw_item:
            if verbose:
                logger.info(f"[SKIP] Invalid item => {raw_item}")
            continue

        # Grab the item directly
        item = raw_item
        prompt = item["prompt"].strip()
        if verbose:
            logger.info(f"\n=== Teacher Prompt {i} ===\n{prompt}")

        responses = []
        teacher_logprobs = []
        rewards_list = []
        skip_this_item = False

        # Generate G responses per item
        for g_idx in range(G):
            resp_text, sum_lp = generate_single_fn(prompt, verbose)

            # Evaluate reward/verifier
            score, feedback = calculate_reward(resp_text, item)
            if score is None:
                # Skip the entire item if reward is None
                if verbose:
                    logger.info(
                        f"[SKIP] item => reward=None at response {g_idx}: {resp_text}"
                    )
                skip_this_item = True
                break

            responses.append(resp_text)
            teacher_logprobs.append(sum_lp)
            rewards_list.append(score)

        if skip_this_item:
            continue  # do not include this item in batch_data

        # Collect data for this item
        batch_data.append({
            "item": item,
            "responses": responses,
            "teacher_logprobs": teacher_logprobs,
            "rewards": rewards_list
        })

    return batch_data
