#!/usr/bin/env python
"""
demo_run_teacher_collection.py

A demo script that uses 'collect_teacher_data_once(...)' for a single-pass 
teacher data collection, skipping invalid or reward=None items automatically.

Key points:
  - We optionally load the teacher model & tokenizer (if you have an actual model_loader).
  - We define an 'integrated_reward' that calls 'combined_calculate_reward' from verifiers.
  - We pass a mock or real 'generate_teacher_fn' to produce teacher outputs.

Any item with missing 'prompt' or a reward of (None,"") is discarded.
"""
# regular imports
import logging

# teacher training imports
from train.teacher.teacher_generation import generate_single_teacher_response
from train.teacher.teacher_model_loader import load_teacher_model
from train.teacher.run_teacher_collection import collect_teacher_data_once

# verifiers
from verifiers.combined_reward import combined_calculate_reward

# info
logging.basicConfig(level=logging.INFO)

def generate_teacher_mock(prompt, verbose=False):
    """
    A mock teacher generation function:
      returns (prompt + " => teacher_out", logprob=1.23).
    If you have a real teacher, replace this with actual generation code.
    """
    response_text = prompt + " => teacher_out"
    if verbose:
        print(f"[GenerateTeacher] Prompt: {prompt}\n => {response_text}")
    return (response_text, 1.23)

def main():
    # set the device to mlx
    device = "mlx"
    teacher_model = "Qwen/Qwen2.5-3B"

    # load the model
    teacher_model, tokenizer, device = load_teacher_model(teacher_model, device_override="cpu")

    # A small dataset, with one item that might be skipped by the reward:
    dataset = [
        {"prompt": "Explain 2+2 step by step."},
        {"prompt": "Write a short poem."},
        {"no_prompt_here": "invalid item => skip inline"},
    ]

    # Single-pass data collection
    final_data = collect_teacher_data_once(
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        dataset=dataset,
        calculate_reward=combined_calculate_reward,
        batch_size=2,
        G=2,
        device=device, 
        verbose=True,
        generate_single_fn=generate_single_teacher_response
    )

    print("\n=== Final Collected Data ===")
    for i, item_data in enumerate(final_data):
        print(f"Item {i} => Original prompt: {item_data['item']['prompt']}")
        print("  Responses:", item_data["responses"])
        print("  Teacher logprobs:", item_data["teacher_logprobs"])
        print("  Rewards:", item_data["rewards"])
        print()

    # If 'combined_calculate_reward' returns None for certain items, or if they lack 'prompt',
    # those items won't appear in final_data.

if __name__ == "__main__":
    main()
