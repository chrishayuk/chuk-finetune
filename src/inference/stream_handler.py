# src/inference/stream_handler.py
from inference.text_utils import strip_role_prefixes, skip_repeated_line

class StreamHandler:
    def __init__(self, known_lines: set):
        self.known_lines = known_lines
        self.first_chunk_printed = False
        self.generated_text_pieces = []

    def process_chunk(self, chunk: str):
        """
        Process a chunk by stripping role prefixes and filtering repeated lines.
        Prints the chunk if it contains new content.
        """
        lines = chunk.splitlines(keepends=True)
        result = []

        for line in lines:
            no_role = strip_role_prefixes(line)
            if skip_repeated_line(no_role, self.known_lines):
                continue
            result.append(no_role)

        cleaned = "".join(result)

        if not self.first_chunk_printed and cleaned.strip():
            print("Assistant: ", end="", flush=True)
            self.first_chunk_printed = True

        print(cleaned, end="", flush=True)
        self.generated_text_pieces.append(chunk)

    def get_full_text(self) -> str:
        return "".join(self.generated_text_pieces)
