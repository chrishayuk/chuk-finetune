#!/usr/bin/env python
"""
student_sft_cli.py

A simple script to fine-tune ("student training") a model using teacher-collected data.

Steps:
  1) Load the student model & tokenizer (Hugging Face).
  2) Load a teacher JSONL dataset (where each line has 'prompt'/'response' or similar).
  3) Convert each item to (input_ids, labels).
  4) Use a standard Trainer (or custom loop) to run supervised fine-tuning.
  5) Save the resulting "student" model to 'output_dir'.
"""
import argparse
import logging

# train imports
from train.teacher.teacher_dataset import TeacherDataset

# import transformers
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    Trainer,
    TrainingArguments,
    DataCollatorWithPadding,
)

# logging
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

def parse_args():
    parser = argparse.ArgumentParser()

    # Core arguments
    parser.add_argument("--student_model_name_or_path", type=str, required=True,
                        help="HuggingFace model name or local path for the student.")
    parser.add_argument("--teacher_data_jsonl", type=str, required=True,
                        help="JSONL file with teacher data (prompt, response).")
    parser.add_argument("--output_dir", type=str, default="./student_output",
                        help="Directory to save the fine-tuned student model.")
    parser.add_argument("--max_length", type=int, default=512,
                        help="Max sequence length for tokenization.")
    parser.add_argument("--batch_size", type=int, default=4,
                        help="Training batch size.")
    parser.add_argument("--num_train_epochs", type=int, default=1,
                        help="Number of training epochs.")
    parser.add_argument("--learning_rate", type=float, default=5e-5,
                        help="Learning rate.")
    parser.add_argument("--save_steps", type=int, default=500,
                        help="Save model every X update steps.")
    parser.add_argument("--logging_steps", type=int, default=100,
                        help="Log training info every X update steps.")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed for reproducibility.")
    
    return parser.parse_args()


def main():
    args = parse_args()

    # 1) Load the tokenizer & model
    logging.info("Loading student tokenizer & model.")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.student_model_name_or_path)

    # 2) Build the dataset
    logging.info("Building teacher dataset.")
    dataset = TeacherDataset(
        jsonl_path=args.teacher_data_jsonl,
        tokenizer=tokenizer,
        max_length=args.max_length
    )

    logging.info(f"Dataset size: {len(dataset)}")

    # 3) HF Training setup
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        evaluation_strategy="no",  # or "steps"/"epoch" if you have a validation set
        seed=args.seed,
    )

    # 4) Create a data collator that pads to the longest sequence in each batch
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer, padding=True)

    # 5) Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=data_collator,  # <-- fix mismatch size by dynamic padding
    )

    # 6) Start fine tuning
    logging.info("Starting student fine-tuning...")
    trainer.train()

    # 7) Save the final model
    logging.info("Saving student model & tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logging.info(f"Done! The student model is saved at '{args.output_dir}'.")

if __name__ == "__main__":
    main()
