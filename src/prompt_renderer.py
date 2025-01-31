from jinja2 import Environment, FileSystemLoader

class PromptRenderer:
    """
    Loads a Jinja template and renders it for dataset questions.
    Supports both:
      - Single formatted string output (default).
      - List of separately formatted prompts (use `as_list=True`).
    """

    @staticmethod
    def render_prompts(questions, template_file='templates/prompt_template.jinja2', as_list=False):
        """
        Loads a Jinja template and renders it with dataset questions.

        :param questions: List of user questions.
        :param template_file: Path to the Jinja2 template file.
        :param as_list: If True, returns a list of formatted prompts (one per question).
                        If False (default), returns a single formatted string.
        :return: Rendered text (either a single string or a list of prompts).
        """
        env = Environment(loader=FileSystemLoader('.'))  # Load from current directory
        template = env.get_template(template_file)

        if as_list:
            # Generate a formatted prompt for **each question separately**
            rendered_prompts = [
                template.render(
                    questions=[question],  # Pass as a list for Jinja processing
                    reasoning_placeholder="Assistant thinks through the problem...",
                    answer_placeholder="Assistant provides the correct answer...",
                    verifier_answer_placeholder="Verifier evaluates the answer..."
                )
                for question in questions
            ]
            return rendered_prompts  # List of formatted prompts

        # Default: Render the template once with all questions
        rendered_text = template.render(
            questions=questions,
            reasoning_placeholder="Assistant thinks through the problem...",
            answer_placeholder="Assistant provides the correct answer...",
            verifier_answer_placeholder="Verifier evaluates the answer..."
        )
        return rendered_text  # Single formatted string