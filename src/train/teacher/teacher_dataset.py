import json
from torch.utils.data import Dataset

class TeacherDataset(Dataset):
    """
    A dataset for finetuning an LLM on prompt+response data.
    
    Format expected per JSON line:
    {
      "item": { "prompt": "..." },  // or top-level "prompt"
      "responses": [...], // array of strings (multiple candidate answers)
      "rewards": [...],   // array of floats, one per response
      "teacher_logprobs": [...], // optional array of floats
      // or you might have a single 'response' if not multiple
    }

    Logic:
      1) If "responses" is a list:
           - If "rewards" is also found and is the same length, pick the best index by highest reward.
             If tie, optionally break tie with highest teacher_logprob.
           - Otherwise, fallback to the *first* response or skip if empty.
      2) If "response" is a string and no "responses" array, use that single answer.
      3) Then we tokenize (prompt + "\n" + chosen_response) and build input_ids, attention_mask, labels.
      4) We mask out the prompt portion in 'labels' by setting them to -100, so only the response part is trained.
    """

    def __init__(self, jsonl_path, tokenizer, max_length=512):
        super().__init__()
        self.samples = []
        self.tokenizer = tokenizer
        self.max_length = max_length

        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                # 1) find the prompt from either item.prompt or top-level
                prompt_text = None
                if "item" in entry and isinstance(entry["item"], dict):
                    prompt_text = entry["item"].get("prompt")
                if prompt_text is None and "prompt" in entry:
                    prompt_text = entry["prompt"]

                if not prompt_text:
                    # skip if no prompt
                    continue

                # 2) check for single 'response' or an array 'responses'
                single_resp = entry.get("response")
                multiple_resps = entry.get("responses")

                chosen_response = None

                if isinstance(multiple_resps, list) and len(multiple_resps) > 0:
                    # we have multiple responses
                    rewards = entry.get("rewards")
                    teacher_logprobs = entry.get("teacher_logprobs")

                    if (isinstance(rewards, list) and len(rewards) == len(multiple_resps)):
                        # pick best idx by highest reward
                        max_reward = max(rewards)
                        # gather all indices that match max_reward
                        best_indices = [i for i, r in enumerate(rewards) if r == max_reward]

                        if len(best_indices) == 1:
                            best_idx = best_indices[0]
                        else:
                            # tie break using teacher_logprobs if available
                            if (isinstance(teacher_logprobs, list) 
                                and len(teacher_logprobs) == len(multiple_resps)):
                                # pick idx in best_indices with highest teacher_logprob
                                best_idx = max(
                                    best_indices,
                                    key=lambda i: teacher_logprobs[i]
                                )
                            else:
                                # no teacher_logprobs or mismatch -> pick first
                                best_idx = best_indices[0]

                        chosen_response = multiple_resps[best_idx]

                    else:
                        # if we have no valid rewards array, fallback to first response
                        chosen_response = multiple_resps[0]

                elif isinstance(single_resp, str):
                    # we have a single response field
                    chosen_response = single_resp

                if not chosen_response:
                    # skip if we never found a valid response
                    continue

                self.samples.append((prompt_text, chosen_response))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        prompt_str, response_str = self.samples[idx]
        # Combine them
        text = f"{prompt_str}\n{response_str}"

        # Tokenize
        encoded = self.tokenizer(
            text,
            max_length=self.max_length,
            truncation=True,
        )
        input_ids = encoded["input_ids"]
        attention_mask = encoded["attention_mask"]
        labels = input_ids.copy()

        # Now we mask out the prompt portion
        # 1) find how many tokens the prompt alone uses
        prompt_encoded = self.tokenizer(
            prompt_str,
            add_special_tokens=False
        )
        prompt_len = len(prompt_encoded["input_ids"])

        # 2) set prompt tokens in labels to -100
        for i in range(min(prompt_len, len(labels))):
            labels[i] = -100

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }
