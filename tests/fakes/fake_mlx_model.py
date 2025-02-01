# tests/fakes/fake_mlx_model.py

import mlx.core as mx
import mlx.nn as nn
import numpy as np

class FakeMLXModel(nn.Module):
    """
    Minimal MLX-like model that:
      1) Inherits from nn.Module.
      2) Has .layers and a .make_cache() so caching won't fail.
      3) Provides a single param to mimic trainable parameters.
      4) Provides a forward(...) method, plus __call__ = forward, so it's callable by MLX code.
    """

    def __init__(self, vocab_size=10, num_layers=1):
        super().__init__()
        self.vocab_size = vocab_size
        # Provide a .layers list so cache.make_prompt_cache(...) won't fail:
        self.layers = [object() for _ in range(num_layers)]

        # A single dummy parameter so MLX can do updates
        self.param = mx.array([0.0], mx.float32)

    def forward(self, tokens: mx.array, cache=None):
        """
        tokens shape: [batch_size=1, seq_len].
        Return shape [1, seq_len, vocab_size].
        We'll also accept 'cache' for incremental decoding (unused).
        """
        batch_size, seq_len = tokens.shape
        # produce random logits
        data = np.random.randn(batch_size, seq_len, self.vocab_size).astype(np.float32)
        return mx.array(data)

    # Make the model callable by referencing forward
    __call__ = forward

    def make_cache(self):
        """
        Return dummy placeholders for each layer.
        """
        return [None] * len(self.layers)

    def parameters(self):
        """
        Return a dict of model parameters for MLX to flatten.
        """
        return {"param": self.param}

    def update(self, new_params: dict):
        """
        MLX's optimizer calls model.update(...) with new params. 
        We'll set self.param to the updated param array.
        """
        self.param = new_params["param"]
