# src/dataset/prompt_renderer.py
from jinja2 import Environment, FileSystemLoader

class PromptRenderer:
    """
    Responsible for loading a Jinja template and creating prompt text from dataset items.

    Supports:
      - Single concatenated output (default).
      - A list of separately rendered prompts (`as_list=True`).
    """

    @staticmethod
    def create_prompt_texts(questions, template_file='templates/prompt_template.jinja2', as_list=False):
        """
        Loads a Jinja template from the specified file and creates prompt text.

        :param questions: List of question strings or question dicts.
        :param template_file: Path to the Jinja2 template file.
        :param as_list: If True, returns a list of separately rendered prompts; 
                        if False (default), returns a single concatenated string.
        :return: Rendered text (a single string or a list of strings).
        """
        env = Environment(loader=FileSystemLoader('.'))  # Load from current directory
        template = env.get_template(template_file)

        if as_list:
            # Generate a prompt for each question in a separate string
            return [
                template.render(
                    questions=[question],
                    reasoning_placeholder="Assistant thinks through the problem...",
                    answer_placeholder="Assistant provides the correct answer...",
                    verifier_answer_placeholder="Verifier evaluates the answer..."
                )
                for question in questions
            ]

        # Default: render one concatenated text with all questions
        return template.render(
            questions=questions,
            reasoning_placeholder="Assistant thinks through the problem...",
            answer_placeholder="Assistant provides the correct answer...",
            verifier_answer_placeholder="Verifier evaluates the answer..."
        )
