# src/cli/train/prompt_handler.py
from cli.train.prompt_renderer import PromptRenderer

def prepare_prompts(dataset):
    """
    Converts each dataset entry into rendered prompt text using the Jinja template.
    Preserves the 'verifiers' key if present.
    
    :param dataset: A list of dataset entries, each containing at least a "prompt" key.
    :return: A list of dicts, each containing:
             - "prompt": the rendered text
             - "verifiers": the original verifiers if present
    """
    prepared = []

    # loop through the dataset
    for item in dataset:
        # Render a single prompt from this dataset entry
        prompt_text = PromptRenderer.create_prompt_texts(
            [item["prompt"]],
            "src/cli/train/templates/prompt_template.jinja2",
            as_list=True
        )[0]

        # create a new entry with the prompt
        new_entry = {
            "prompt": prompt_text
        }

        # If the item contains 'verifiers', include them
        if "verifiers" in item:
            new_entry["verifiers"] = item["verifiers"]

        # add the entry
        prepared.append(new_entry)

    # return the prepared prompts
    return prepared
