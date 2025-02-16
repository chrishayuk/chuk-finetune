# src/train/teacher/teacher_data_prepare.py
import logging
from typing import Any, List, Dict, Optional, Tuple, Callable

logger = logging.getLogger(__name__)

def prepare_batch_data_for_teacher(
    batch_questions: List[Any],
    generate_single_fn: Callable[[str, bool], Tuple[str, float]],
    calculate_reward: Callable[[str, Dict[str, Any]], Tuple[Optional[float], str]],
    G: int = 4,
    verbose: bool = False,
    prompt_template: Optional[str] = None
) -> List[Dict[str, Any]]:
    """
    A teacher-specific data-preparation function that:
      1) Reads an optional 'min_reward' field from each dataset item.
      2) (Optional) Applies 'prompt_template' if provided ({{question}} -> original prompt).
      3) Generates G responses per valid item, computing rewards.
      4) Skips items if any response's reward is None OR if *no* response meets 'min_reward'.
      5) Returns a list of data dicts with "item", "responses", "teacher_logprobs",
         "rewards", and "feedbacks".

    For per-item min_reward usage:
    - If raw_item["min_reward"] is given, we enforce that at least one response
      must be >= that min_reward. Otherwise we skip the item.
    - If "min_reward" is missing, default to 1.0 (or another fallback).

    Returns:
        A list of dicts. Each dict looks like:
        {
            "item": {...},  # includes the final 'prompt' or 'templated_prompt',
            "responses": [...],
            "teacher_logprobs": [...],
            "rewards": [...],
            "feedbacks": [...],
        }
    """
    batch_data = []

    for i, raw_item in enumerate(batch_questions):
        # 1) Must be a dict containing 'prompt'
        if not isinstance(raw_item, dict) or "prompt" not in raw_item:
            if verbose:
                logger.info(f"[SKIP] Invalid item => {raw_item}")
            continue

        # 2) If a prompt_template is provided, apply it
        original_prompt = raw_item["prompt"]
        if prompt_template:
            transformed_prompt = prompt_template.replace("{{question}}", original_prompt)
            raw_item["prompt"] = transformed_prompt
        else:
            # Use the original prompt as is
            raw_item["prompt"] = original_prompt

        # Store original prompt for reference if you'd like
        # raw_item["prompt_original"] = original_prompt

        # 3) Grab (or default) the per-item min_reward
        #    If the item doesn't define "min_reward", fallback to 1.0
        item_min_reward = raw_item.get("min_reward", 1.0)

        # Prepare to collect responses
        item = raw_item
        prompt = item["prompt"].strip()

        if verbose:
            logger.info(f"\n=== Teacher Prompt {i} ===\n{prompt}")

        responses = []
        teacher_logprobs = []
        rewards_list = []
        feedback_list = []
        skip_this_item = False

        # Generate G responses per item
        for g_idx in range(G):
            resp_text, sum_lp = generate_single_fn(prompt, verbose)

            # Evaluate reward
            score, feedback = calculate_reward(resp_text, item)

            # If reward is None, skip the entire item
            if score is None:
                if verbose:
                    logger.info(
                        f"[SKIP] item => reward=None at response {g_idx}: {resp_text}"
                    )
                skip_this_item = True
                break

            # Collect data for this response
            responses.append(resp_text)
            teacher_logprobs.append(sum_lp)
            rewards_list.append(score)
            feedback_list.append(feedback)

        # If any response was None => skip
        if skip_this_item:
            continue

        # 4) Check if this item meets its min_reward:
        #    If *none* of the G responses >= item_min_reward, skip the item
        if not any(r >= item_min_reward for r in rewards_list):
            if verbose:
                logger.info(
                    f"[SKIP] item => no response met min_reward={item_min_reward}, "
                    f"rewards={rewards_list}"
                )
            continue

        # 5) Keep the item if it passed all checks
        batch_data.append({
            "item": item,
            "responses": responses,
            "teacher_logprobs": teacher_logprobs,
            "rewards": rewards_list,
            "feedbacks": feedback_list,
        })

    return batch_data