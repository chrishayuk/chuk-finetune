# src/cli/train/prompt_handler.py
import os

# imports
from cli.train.prompt_renderer import PromptRenderer

def prepare_prompts(dataset, template_path=None):
    """
    Converts each dataset entry into rendered prompt text using the Jinja template.
    Preserves the 'verifiers' key if present.
    
    :param dataset: A list of dataset entries, each containing at least a "prompt" key.
    :param template_path: Optional path to the Jinja template. If None, defaults
                          to 'templates/prompt_template.jinja2' in the same
                          folder as this file.
    :return: A list of dicts, each containing:
             - "prompt": the rendered text
             - "verifiers": the original verifiers if present
    """
    # If no custom template_path is provided, use the local templates directory
    if template_path is None:
        base_dir = os.path.dirname(__file__)
        template_path = os.path.join(base_dir, "templates", "prompt_template.jinja2")

    prepared = []

    for item in dataset:
        # Render a single prompt from this dataset entry
        prompt_text = PromptRenderer.create_prompt_texts(
            [item["prompt"]],
            template_path,
            as_list=True
        )[0]

        # Create a new entry with the prompt
        new_entry = {
            "prompt": prompt_text
        }

        # If the item contains 'verifiers', include them
        if "verifiers" in item:
            new_entry["verifiers"] = item["verifiers"]

        # Add to the prepared prompts list
        prepared.append(new_entry)

    return prepared
