# src/dataset_loader.py
import json

# imports
from cli.train.logger_config import logger, color_text, BOLD

def load_dataset(jsonl_path="dataset/new_completions.jsonl"):
    """
    Loads a dataset from a JSON Lines file. Each line in the file should contain a valid JSON object with:
      - "prompt": the text prompt to give the model.
      
      And EITHER:
      - "verifiers": a list of dicts, where each dict includes:
            - "name": the name of the verifier (matching your JSON registry)
            - "url": the verifier URL (defaulting to http://0.0.0.0:8000 if not provided)
      
      OR:
      - "completion": the expected output, used for a standard SFT check.
      
    :param jsonl_path: The path to the JSONL file containing the dataset.
    :return: A list of dictionaries, each representing a dataset entry.
    """
    # empty dataset
    dataset = []

    try:
        # open the file
        with open(jsonl_path, 'r', encoding='utf-8') as file:
            # loop through each line
            for line in file:
                # strip out spaces
                line = line.strip()

                # skip empty lines
                if not line:
                    continue

                try:
                    # load the json line
                    record = json.loads(line)

                    # add to the dataset
                    dataset.append(record)
                except json.JSONDecodeError as e:
                    # skip line
                    logger.warning(f"Skipping line due to JSON decode error: {e}")
    except FileNotFoundError:
        # file not found
        logger.error(f"File not found: {jsonl_path}")
    
    return dataset

if __name__ == "__main__":
    # load the dataset
    data = load_dataset()
    
    # log it
    logger.info(f"Loaded {len(data)} records from the dataset.")
