# tests/fakes/fake_torch_model.py
import torch

class FakeHFOutput:
    """
    Mimics a Hugging Face model output with .logits
    """
    def __init__(self, logits: torch.Tensor):
        self.logits = logits

class FakeTorchModel(torch.nn.Module):
    def __init__(self, vocab_size=10):
        super().__init__()
        self.vocab_size = vocab_size
        # A trivial linear to produce logits
        self.linear = torch.nn.Linear(5, vocab_size, bias=False)

    def forward(self, input_ids):
        """
        input_ids: shape [batch_size, seq_len]
        Return an object with .logits => [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len = input_ids.shape
        # contrived embeddings
        embedded = torch.randn(batch_size, seq_len, 5, device=input_ids.device)
        out = self.linear(embedded.view(batch_size*seq_len, 5))
        logits = out.view(batch_size, seq_len, self.vocab_size)
        # Return a 'FakeHFOutput' with .logits
        return FakeHFOutput(logits)

    def generate(self, **kwargs):
        """
        Return G sequences of shape [seq_len], each containing random IDs in [0, vocab_size).
        """
        num_return_sequences = kwargs.get("num_return_sequences", 1)
        max_length = kwargs.get("max_length", 3)
        device = next(self.parameters()).device

        # produce random IDs => [num_return_sequences, max_length]
        out = torch.randint(
            low=0, high=self.vocab_size,
            size=(num_return_sequences, max_length),
            device=device
        )
        return out
