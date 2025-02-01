# tests/fakes/fake_mlx_tokenizer.py
class FakeMLXTokenizer:
    """
    Minimal MLX-like tokenizer with .encode(...), .decode(...),
    bos_token, eos_token_id, and other HF-like attributes.
    """
    def __init__(self, vocab_size=10):
        self.vocab_size = vocab_size
        self.bos_token = None
        self.eos_token = None
        self.eos_token_id = vocab_size - 1

        # MLX-lm expects this attribute:
        self.clean_up_tokenization_spaces = True

    def encode(self, text: str, add_special_tokens: bool = False):
        import numpy as np
        return np.random.randint(0, self.vocab_size, size=3).tolist()

    def decode(self, token_ids, skip_special_tokens=True):
        if not isinstance(token_ids, (list, tuple)):
            token_ids = token_ids.tolist()
        return " ".join(f"tok{i}" for i in token_ids)
