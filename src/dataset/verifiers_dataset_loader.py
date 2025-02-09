# src/dataset/verifiers_dataset_loader.py
import json
from cli.train.logger_config import logger

def load_prompts_and_verifiers(jsonl_path="dataset/zero/math_very_easy.jsonl"):
    """
    Loads a dataset of prompt-verifier pairs from a JSON Lines file.
    Each line should include:
      - "prompt": the text prompt
      - "verifiers": a list of verifier dicts (optional, defaults to empty)

    :param jsonl_path: The path to the JSONL file containing the dataset.
    :return: A list of dicts, each including "prompt" and "verifiers".
    """
    dataset = []

    try:
        # open the file
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            # loop through line
            for line in file:
                # get the line
                line = line.strip()
                
                # if there is no line, skip on
                if not line:
                    continue

                try:
                    # load the json
                    record = json.loads(line)

                    # Normalise the record to a dict with "prompt" and "verifiers"
                    prompt = record.get("prompt", "")
                    verifiers = record.get("verifiers", [])

                    # add to the dataset
                    dataset.append({
                        "prompt": prompt,
                        "verifiers": verifiers
                    })

                except json.JSONDecodeError as e:
                    # error
                    logger.warning(f"Skipping line due to JSON decode error: {e}")

    except FileNotFoundError:
        logger.error(f"File not found: {jsonl_path}")

    return dataset

if __name__ == "__main__":
    # load the dataset
    data = load_prompts_and_verifiers()

    # loaded
    logger.info(f"Loaded {len(data)} records with prompt-verifier structure.")
