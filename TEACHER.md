## Collecting data from a teacher
The following will collect data from a teacher model and save to a jsonl file for training a student later.

```bash
python teacher_collection_cli.py --model "Qwen/Qwen2.5-3B" --dataset dataset/teacher/input.jsonl --output dataset/teacher/output.jsonl --device cpu --batch_size 2 --G 4
```

## Student Fine Tuning
The following will take the outputted data from a teacher model and train on that data

```bash
python student_sft_train_cli.py \
    --student_model_name_or_path "Qwen/Qwen2.5-3B" \
    --teacher_data_jsonl "dataset/teacher/output.jsonl" \
    --output_dir "./student_output" \
    --num_train_epochs 2 \
    --batch_size 2 \
    --max_length 512
```

## SFT Fine Tuning
The following will take regular prompt/completion data and perform a regular fine tune

```bash
python sft_train_cli.py \
  --model_name_or_path "Qwen/Qwen2.5-3B" \
  --sft_data_jsonl "dataset/coldstart/math_completions.jsonl" \
  --output_dir "./sft_output" \
  --num_train_epochs 2 \
  --batch_size 2 \
  --max_length 512 \
  --learning_rate 5e-5
```

## Infer
The following will infer the saved model

```bash
python infer_student.py --model_path "./student_output" --prompt "Calculate 50 - 123."
```

or

```bash
python infer_student.py --model_path "./sft_output" --prompt "Calculate 50 - 123."
```