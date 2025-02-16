import json
import logging
import torch
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

class SFTDataset(Dataset):
    """
    A straightforward SFT dataset where each JSON line has:
      {
        "prompt": "...",
        "completion": "..."
      }

    This dataset:
      1) Concatenates prompt + completion into one string,
      2) Tokenizes it (returning PyTorch Tensors),
      3) Sets labels = copy of input_ids,
      4) Masks out the prompt portion in labels by setting them to -100.

    This updated version includes debug checks and returns lists (not tensors) so that
    the collator can pad them correctly.
    """

    def __init__(self, jsonl_path, tokenizer, max_length=512):
        super().__init__()
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    logger.warning(f"[line {line_num}] Skipping: JSON parse error.")
                    continue

                prompt_text = entry.get("prompt")
                completion_text = entry.get("completion")

                # Skip if either field is missing or not a string
                if not prompt_text or not completion_text:
                    logger.warning(f"[line {line_num}] Skipping: Missing prompt/completion.")
                    continue
                if not isinstance(prompt_text, str) or not isinstance(completion_text, str):
                    logger.warning(f"[line {line_num}] Skipping: Non-string prompt/completion.")
                    continue

                self.samples.append({
                    "line_num": line_num,
                    "prompt": prompt_text,
                    "completion": completion_text
                })

        logger.info(f"Loaded {len(self.samples)} valid samples from {jsonl_path}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        example = self.samples[idx]
        line_num = example["line_num"]
        prompt_str = example["prompt"]
        completion_str = example["completion"]

        # 1) Combine into one string
        text = f"{prompt_str}{completion_str}"

        # 2) Tokenize full text (returns tensors of shape [1, seq_len])
        tokenized = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
            return_tensors="pt"
        )
        if "input_ids" not in tokenized or "attention_mask" not in tokenized:
            logger.warning(
                f"[line {line_num}] Skipping: Missing 'input_ids' or 'attention_mask' in tokenizer output."
            )
            raise ValueError(f"[line {line_num}] Tokenizer output missing keys.")
            
        input_ids = tokenized["input_ids"]
        attention_mask = tokenized["attention_mask"]

        # Debug check: Expect 2D tensors of shape [1, seq_len]
        if input_ids.ndim != 2 or attention_mask.ndim != 2:
            logger.warning(
                f"[line {line_num}] Unexpected shape: input_ids={list(input_ids.shape)}, attention_mask={list(attention_mask.shape)}"
            )
            raise ValueError(f"[line {line_num}] Tokenized output shape invalid.")

        # Convert from [1, seq_len] to 1D tensor
        input_ids = input_ids[0]
        attention_mask = attention_mask[0]
        labels = input_ids.clone()

        # 3) Tokenize the prompt alone to determine its token length
        prompt_tokenized = self.tokenizer(
            prompt_str,
            add_special_tokens=False,
            return_tensors="pt"
        )
        if "input_ids" not in prompt_tokenized:
            logger.warning(f"[line {line_num}] Skipping: Prompt tokenizer missing 'input_ids'.")
            raise ValueError(f"[line {line_num}] Prompt tokenizer output invalid.")

        prompt_len = prompt_tokenized["input_ids"].shape[1]

        # 4) Mask out prompt tokens in labels
        if prompt_len > labels.size(0):
            logger.warning(
                f"[line {line_num}] Prompt length ({prompt_len}) exceeds total tokens ({labels.size(0)}). Clamping."
            )
            prompt_len = labels.size(0)
        labels[:prompt_len] = -100

        # Final debug check: ensure shapes match
        if input_ids.size(0) != labels.size(0):
            logger.warning(
                f"[line {line_num}] Mismatch: input_ids length {input_ids.size(0)} vs labels length {labels.size(0)}"
            )
            raise ValueError(f"[line {line_num}] Token length mismatch.")

        # Convert tensors to lists so that the collator can pad properly.
        sample = {
            "input_ids": input_ids.tolist(),
            "attention_mask": attention_mask.tolist(),
            "labels": labels.tolist()
        }

        logger.debug(f"[line {line_num}] Final sample length: {len(sample['input_ids'])} tokens.")
        return sample

def custom_data_collator(features, tokenizer, pad_token_id=None):
    """
    Expects features to be a list of dicts with keys: "input_ids", "attention_mask", "labels".
    Pads each sequence to the maximum length in the batch.
    """
    if pad_token_id is None:
        pad_token_id = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else 0

    # Determine max length in this batch
    max_length = max(len(feature["input_ids"]) for feature in features)

    input_ids_list = []
    attention_mask_list = []
    labels_list = []

    for feature in features:
        input_ids = feature["input_ids"]
        attention_mask = feature["attention_mask"]
        labels = feature["labels"]
        pad_len = max_length - len(input_ids)

        # Pad input_ids with pad_token_id, attention_mask with 0, and labels with -100
        input_ids_padded = input_ids + [pad_token_id] * pad_len
        attention_mask_padded = attention_mask + [0] * pad_len
        labels_padded = labels + [-100] * pad_len

        input_ids_list.append(input_ids_padded)
        attention_mask_list.append(attention_mask_padded)
        labels_list.append(labels_padded)

    batch = {
        "input_ids": torch.tensor(input_ids_list, dtype=torch.long),
        "attention_mask": torch.tensor(attention_mask_list, dtype=torch.long),
        "labels": torch.tensor(labels_list, dtype=torch.long)
    }
    return batch
