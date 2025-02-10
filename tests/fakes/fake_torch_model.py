# tests/fakes/fake_torch_model.py

import torch

class FakeHFOutput:
    """
    Mimics a Hugging Face model output with .logits and optional .past_key_values
    """
    def __init__(self, logits: torch.Tensor, past_key_values=None):
        self.logits = logits
        self.past_key_values = past_key_values

class FakeTorchModel(torch.nn.Module):
    def __init__(self, vocab_size=10):
        super().__init__()
        self.vocab_size = vocab_size
        # A trivial linear to produce logits
        self.linear = torch.nn.Linear(5, vocab_size, bias=False)

    def forward(self, input_ids, use_cache=None, **kwargs):
        """
        input_ids: shape [batch_size, seq_len]
        Return a 'FakeHFOutput' with .logits => [batch_size, seq_len, vocab_size].
        Accepts use_cache and any additional HF-like kwargs, ignoring them.
        """
        batch_size, seq_len = input_ids.shape

        # Contrived embeddings: [batch_size, seq_len, 5]
        embedded = torch.randn(batch_size, seq_len, 5, device=input_ids.device)

        # Flatten => [batch_size*seq_len, 5]
        out = self.linear(embedded.view(batch_size * seq_len, 5))  # => [B*T, vocab_size]

        # Reshape back to [batch_size, seq_len, vocab_size]
        logits = out.view(batch_size, seq_len, self.vocab_size)

        # Return an object mimicking a HF model (including .past_key_values)
        return FakeHFOutput(logits=logits, past_key_values=None)

    def generate(self, **kwargs):
        """
        Return random sequences of IDs in [0, vocab_size).
        Mimics the HF generate method signature minimally.
        """
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        max_length = kwargs.get("max_length", 3)
        device = next(self.parameters()).device

        # Produce random IDs => shape [num_return_sequences, max_length]
        return torch.randint(
            low=0,
            high=self.vocab_size,
            size=(num_return_sequences, max_length),
            device=device
        )
