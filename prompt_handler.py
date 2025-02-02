# prompt_handler.py
from prompt_renderer import PromptRenderer

def render_prompts(dataset):
    rendered = []
    for item in dataset:
        prompt_text = PromptRenderer.render_prompts(
            [item["prompt"]],
            "src/templates/prompt_template.jinja2",
            as_list=True
        )[0]
        rendered.append({
            "prompt": prompt_text,
            "verifier": item.get("verifier", "haiku"),
            "verifier_url": item.get("verifier_url", "http://0.0.0.0:8000")
        })
    return rendered
