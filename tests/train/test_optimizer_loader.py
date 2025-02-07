# tests/test_optimizer_loader.py
import pytest

from train.optimizer_loader import get_optimizer

@pytest.mark.parametrize("framework", ["torch", "mlx"])
def test_get_optimizer_known_frameworks(framework):
    """
    Test that get_optimizer returns the correct optimizer instance for
    known frameworks: "torch" and "mlx".
    """
    if framework == "torch":
        # We need a minimal Torch model to pass to get_optimizer
        import torch
        import torch.nn as nn
        from torch.optim import AdamW

        class DummyTorchModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(2, 1)

        model = DummyTorchModel()
        lr = 1e-3
        optimizer = get_optimizer(framework, model, lr)
        assert isinstance(optimizer, AdamW), "Expected torch AdamW optimizer"

        # Optionally check that it has the correct LR
        for param_group in optimizer.param_groups:
            assert param_group["lr"] == pytest.approx(lr)

    elif framework == "mlx":
        # We'll assume MLX is installed and import it
        import mlx.optimizers as optim

        # We don't strictly need a model for MLX's Adam, but let's do a quick check
        class DummyMLXModel:
            def __init__(self):
                pass

        model = DummyMLXModel()
        lr = 1e-3
        optimizer = get_optimizer(framework, model, lr)
        # For MLX, we expect `optim.Adam`
        assert isinstance(optimizer, optim.Adam), "Expected MLX Adam optimizer"
        # If needed, you might also check internal attributes, if they exist
        # (but typically MLX's Adam might not expose a direct 'lr' attribute).


def test_get_optimizer_unknown():
    """
    Test that an unknown framework raises ValueError.
    """
    with pytest.raises(ValueError, match="Unknown framework"):
        get_optimizer("unknown_framework", None, 1e-3)
