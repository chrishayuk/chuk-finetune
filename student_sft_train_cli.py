#!/usr/bin/env python
"""
student_sft_train_cli.py
A script to fine-tune ("student training") a model using teacher-collected data.
"""
import argparse
import logging
import torch

# train imports
from train.teacher.teacher_dataset import TeacherDataset

# import transformers
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

    parser.add_argument("--student_model_name_or_path", type=str, required=True)
    parser.add_argument("--teacher_data_jsonl", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="./student_output")
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=5e-5)
    parser.add_argument("--save_steps", type=int, default=500)
    parser.add_argument("--logging_steps", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    
    return parser.parse_args()


def custom_collator(features, tokenizer):
    """
    We expect each sample in `features` to be a dict with:
      - 'input_ids': list[int]
      - 'attention_mask': list[int]
      - 'labels': list[int]  (already masked out prompt tokens w/ -100)
    This collator will pad them consistently.
    """
    # Separate out each field
    input_ids_list = [f["input_ids"] for f in features]
    attention_mask_list = [f["attention_mask"] for f in features]
    labels_list = [f["labels"] for f in features]

    # Use the tokenizer's pad to pad 'input_ids' & 'attention_mask'
    padded_inputs = tokenizer.pad(
        {"input_ids": input_ids_list, "attention_mask": attention_mask_list},
        padding=True,
        return_tensors="pt",
    )

    # Manually pad 'labels' to the same max length, using -100
    max_label_len = max(len(labels) for labels in labels_list)
    padded_labels = []
    for labels in labels_list:
        needed = max_label_len - len(labels)
        padded_labels.append(labels + ([-100] * needed))

    # Convert to tensor
    padded_labels = torch.tensor(padded_labels, dtype=torch.long)

    return {
        "input_ids": padded_inputs["input_ids"],
        "attention_mask": padded_inputs["attention_mask"],
        "labels": padded_labels,
    }


def main():
    args = parse_args()

    # 1) Load tokenizer & model
    logger.info("Loading student tokenizer & model.")
    tokenizer = AutoTokenizer.from_pretrained(args.student_model_name_or_path)
    model = AutoModelForCausalLM.from_pretrained(args.student_model_name_or_path)

    # Ensure we have a pad token
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id

    # 2) Teacher dataset
    logger.info("Building teacher dataset.")
    dataset = TeacherDataset(
        jsonl_path=args.teacher_data_jsonl,
        tokenizer=tokenizer,
        max_length=args.max_length
    )
    logger.info(f"Dataset size: {len(dataset)}")

    # 3) TrainingArguments
    training_args = TrainingArguments(
        output_dir=args.output_dir,
        overwrite_output_dir=True,
        num_train_epochs=args.num_train_epochs,
        per_device_train_batch_size=args.batch_size,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        evaluation_strategy="no",  
        seed=args.seed,
    )

    # 4) Use custom collator
    def collator_fn(batch):
        return custom_collator(batch, tokenizer)

    # 5) Initialize Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=dataset,
        data_collator=collator_fn,
    )

    # 6) Train
    logger.info("Starting student fine-tuning...")
    trainer.train()

    # 7) Save final model
    logger.info("Saving student model & tokenizer...")
    trainer.save_model(args.output_dir)
    tokenizer.save_pretrained(args.output_dir)

    logger.info(f"Done! The student model is saved at '{args.output_dir}'.")

if __name__ == "__main__":
    main()