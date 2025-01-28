import numpy as np
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer
)
from sklearn.metrics import accuracy_score

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return {"accuracy": accuracy_score(labels, predictions)}

def main():
    # Load a small dataset for demonstration
    dataset = load_dataset("glue", "mrpc")

    # Download a pre-trained tokenizer
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")

    # Tokenise the dataset
    def tokenize_function(example):
        return tokenizer(
            example["sentence1"],
            example["sentence2"],
            truncation=True,
            padding="max_length",
            max_length=128
        )

    tokenised_dataset = dataset.map(tokenize_function, batched=True)

    # Prepare for PyTorch training
    tokenised_dataset = tokenised_dataset.remove_columns(["sentence1", "sentence2", "idx"])
    tokenised_dataset = tokenised_dataset.rename_column("label", "labels")
    tokenised_dataset.set_format("torch")

    train_dataset = tokenised_dataset["train"]
    eval_dataset = tokenised_dataset["validation"]

    # Load a pre-trained model for sequence classification
    model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2)

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        save_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=1,
        weight_decay=0.01,
        logging_steps=10,
        logging_dir="./logs"
    )

    # Define the Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics
    )

    # Train the model
    trainer.train()

    # Evaluate the model
    eval_results = trainer.evaluate()
    print("Evaluation results:", eval_results)

if __name__ == "__main__":
    main()
