# tests/fakes/fake_torch_tokenizer.py
import torch

class FakeTorchTokenizer:
    def __init__(self, vocab_size=10):
        self.vocab_size = vocab_size

    def __call__(self, text, return_tensors=None):
        """
        Return a small 'input_ids' tensor. 
        We'll just produce length=3 for every string, 
        with random IDs in [0, vocab_size).
        """
        # Not a real parser, just a random generator
        rng = torch.Generator()
        # so the result is reproducible, or you can skip setting a seed
        tokens = torch.randint(
            low=0, high=self.vocab_size,
            size=(1, 3), dtype=torch.long
        )
        if return_tensors == "pt":
            return {"input_ids": tokens}
        else:
            return tokens.tolist()

    def decode(self, token_ids, skip_special_tokens=True):
        """
        Turn e.g. [1,5,2] into a mock string, e.g. 'tok1 tok5 tok2'.
        We'll do something naive:
        """
        if isinstance(token_ids, torch.Tensor):
            token_ids = token_ids.tolist()
        return " ".join(f"tok{i}" for i in token_ids)

