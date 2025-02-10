# tests/fakes/fake_torch_tokenizer.py
import torch

class FakeTorchTokenizer:
    def __init__(self, vocab_size=10):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.eos_token_id = 1

    def __call__(
        self,
        text,
        return_tensors=None,
        padding=False,
        truncation=False,
        max_length=None,
        **kwargs
    ):
        """
        Mimic the HF tokenizer's __call__, returning a dict with 'input_ids'.
        Accepts `padding`, `truncation`, `max_length`, ignoring them unless needed.
        """
        # 1) Handle single string vs. list of strings
        if isinstance(text, str):
            tokenized_batch = [self.encode(text)]
        else:
            tokenized_batch = [self.encode(t) for t in text]

        # 2) If truncation is True and max_length is set, truncate each sequence
        if truncation and max_length is not None:
            tokenized_batch = [
                tokens[:max_length] for tokens in tokenized_batch
            ]

        # 3) If padding is True, pad all sequences to the longest length
        if padding is True:
            max_len = max(len(tokens) for tokens in tokenized_batch)
            for i, tokens in enumerate(tokenized_batch):
                pad_len = max_len - len(tokens)
                # Pad with self.pad_token_id
                tokenized_batch[i] = tokens + [self.pad_token_id] * pad_len

        # 4) Convert to tensor if return_tensors="pt"
        if return_tensors == "pt":
            return {
                "input_ids": torch.tensor(tokenized_batch, dtype=torch.long)
            }
        else:
            # Return plain list
            return {"input_ids": tokenized_batch}

    def encode(self, text):
        """
        A minimal encode method that returns a list of token IDs (in [0, vocab_size)).
        Here, we just do a naive "word -> integer" approach.
        """
        tokens = []
        for w in text.split():
            # Hash word, clamp to [0, vocab_size)
            tok_id = abs(hash(w)) % self.vocab_size
            tokens.append(tok_id)
        return tokens

    def decode(self, token_ids):
        """
        Minimal decode that just prints a token placeholder.
        """
        return " ".join(f"<tok{tid}>" for tid in token_ids)

    @property
    def bos_token_id(self):
        return 2
