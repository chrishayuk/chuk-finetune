#!/usr/bin/env python

"""
demo_run_teacher_collection.py

A demo script that uses 'collect_teacher_data_once(...)' for a single-pass 
teacher data collection, skipping invalid or reward=None items automatically.
"""

import logging
from train.teacher.run_teacher_collection import collect_teacher_data_once

logging.basicConfig(level=logging.INFO)

def generate_teacher_mock(prompt, verbose=False):
    response_text = prompt + " => teacher_out"
    if verbose:
        print(f"[GenerateTeacher] Prompt: {prompt}\n => {response_text}")
    return (response_text, 1.23)

def calculate_reward_mock(resp_text, item):
    if "skip" in item["prompt"].lower():
        return (None, "")
    return (1.0, "")

def main():
    teacher_model = "mock_teacher_model"
    tokenizer = "mock_tokenizer"

    dataset = [
        {"prompt": "Explain 2+2 step by step."},
        {"prompt": "Please skip this prompt."},
        {"prompt": "Write a short poem."},
        {"some_other_key": "no prompt => skip automatically"},
    ]

    final_data = collect_teacher_data_once(
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        dataset=dataset,
        calculate_reward=calculate_reward_mock,
        batch_size=2,
        G=2,
        device="cpu",
        verbose=True,

        # Optionally pass a custom generation function
        generate_single_fn=generate_teacher_mock
    )

    print("\n=== Final Collected Data ===")
    for i, item_data in enumerate(final_data):
        print(f"Item {i} => Original prompt: {item_data['item']['prompt']}")
        print("  Responses:", item_data["responses"])
        print("  Teacher logprobs:", item_data["teacher_logprobs"])
        print("  Rewards:", item_data["rewards"])
        print()

if __name__ == "__main__":
    main()
