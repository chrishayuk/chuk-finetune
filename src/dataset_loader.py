# src/dataset_loader.py
def load_dataset():
    """
    Returns a list of dicts. Each dict has:
      - "prompt": the text prompt to give the model
      - "verifier": the name of the verifier (matching your JSON registry)
      - "verifier_url": defaulting to http://0.0.0.0:8000
    """
    print("[INFO] Loading dataset (poetry prompts with verifiers)...")
    dataset = [
        {
            "prompt": "Write a haiku about the beauty of nature in spring.",
            "verifier": "haiku",
            "verifier_url": "http://0.0.0.0:8000"
        },
        {
            "prompt": "Please compose a limerick about a programmer who found a bug in their code.",
            "verifier": "limerick",
            "verifier_url": "http://0.0.0.0:8000"
        },
        {
            "prompt": "Give me a short poem with strong end rhymes about cats playing with yarn.",
            "verifier": "rhyme",
            "verifier_url": "http://0.0.0.0:8000"
        },
        {
            "prompt": "Create a tanka that explores the changing seasons, from winter’s cold to spring’s warmth.",
            "verifier": "tanka",
            "verifier_url": "http://0.0.0.0:8000"
        }
    ]

    # return the dataset
    return dataset
