# src/train/grpo/grpo_prepare.py
from typing import Any, Dict, Callable, List, Optional, Tuple

def prepare_batch_data_for_grpo(
    batch_questions: List[Any],
    ensure_dict_fn: Callable[[Any], Optional[Dict[str, Any]]],
    generate_single_fn: Callable[[str, bool], Tuple[str, float]],
    calculate_reward: Callable[[str, Dict[str, Any]], Tuple[Optional[float], str]],
    G: int = 4,
    verbose: bool = False
) -> List[Dict[str, Any]]:
    """
    A shared function for preparing batch data for GRPO training.
    It loops over raw items, ensures they are valid via 'ensure_dict_fn',
    then for each item it calls 'generate_single_response_fn' G times 
    (one response at a time), collects old_logprob, and calls 'calculate_reward'.
    
    If 'calculate_reward' returns None, we skip that entire item.
    
    Args:
        batch_questions: A list of raw items (dicts, strings, etc.).
        ensure_dict_fn: A function that tries to parse/convert raw items into
            a valid dict with at least {"prompt": ...}.
        generate_single_response_fn: A function that takes (prompt, verbose)
            and returns (response_text, old_logprob) as (str, float).
        calculate_reward: A function (resp_text, item_dict) => (score: Optional[float], feedback: str).
        G: number of responses to generate per item.
        verbose: if True, prints or logs debug info.

    Returns:
        A list of data dicts ready for GRPO:
          [
            {
              "item": <the original item dict>,
              "responses": [resp_1, resp_2, ...],
              "old_logprobs": [lp_1, lp_2, ...],
              "rewards": [r_1, r_2, ...],
            },
            ...
          ]
    """
    batch_data = []

    for i, raw_item in enumerate(batch_questions):
        item = ensure_dict_fn(raw_item)
        if item is None:
            if verbose:
                print(f"[SKIP] Invalid item => {raw_item}")
            continue

        prompt = item["prompt"].strip()
        if verbose:
            print(f"\n=== Prompt {i} ===\n{prompt}")

        responses = []
        old_logprobs = []
        rewards_list = []
        skip_this_item = False

        # Generate G responses
        for g_idx in range(G):
            resp_text, sum_lp = generate_single_fn(prompt, verbose)
            
            # Calculate reward
            score, feedback_text = calculate_reward(resp_text, item)
            if score is None:
                # skip entire item
                if verbose:
                    print(f"[SKIP] item due to None reward for response {g_idx}: {resp_text}")
                skip_this_item = True
                break

            responses.append(resp_text)
            old_logprobs.append(sum_lp)
            rewards_list.append(score)

        if skip_this_item:
            continue

        batch_data.append({
            "item": item,
            "responses": responses,
            "old_logprobs": old_logprobs,
            "rewards": rewards_list,
        })

    return batch_data
