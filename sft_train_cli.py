#!/usr/bin/env python
"""
sft_train_cli.py

A regular SFT training script that:
  1) Loads a base (student) model,
  2) Reads a JSONL with prompt+completion lines,
  3) Builds SFTDataset & a custom data collator,
  4) Finetunes the model with the Trainer,
  5) Saves the model to 'output_dir'.
"""

import argparse
import logging
import json
import os
import torch
from train.sft.sft_dataset import SFTDataset, custom_data_collator
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name_or_path", type=str, required=True)
    parser.add_argument("--sft_data_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./sft_output")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args()

def main():
    args = parse_args()
    logger.info(f"Loading base model: {args.model_name_or_path}")

    # For Qwen, you might need trust_remote_code=True; add if needed.
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        trust_remote_code=True
    )

    if torch.cuda.is_available():
        model.cuda()

    logger.info("Building SFT dataset")
    dataset = SFTDataset(
        jsonl_path=args.sft_data_jsonl,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    logger.info(f"Dataset size: {len(dataset)}")

    # Create a lambda wrapper for the custom collator so it can access the tokenizer
    data_collator = lambda features: custom_data_collator(features, tokenizer)

    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        logging_steps=50,
        save_steps=200,
        seed=args.seed,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,
    )

    logger.info("Starting finetuning...")
    trainer.train()

    logger.info("Saving model & tokenizer")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info("Done!")

if __name__ == "__main__":
    main()