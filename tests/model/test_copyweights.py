# test_copy_weights.py

import pytest
import torch
import numpy as np

try:
    import mlx.core as mx
    from mlx.utils import tree_flatten
    HAS_MLX = True
except ImportError:
    HAS_MLX = False

from model.copy_weights import copy_weights

# ---------------------------------------------------------------------
# 1) Define minimal Torch & MLX model classes
# ---------------------------------------------------------------------

class SimpleTorchModel(torch.nn.Module):
    """A minimal PyTorch model with a single linear layer."""
    def __init__(self, in_dim=3, out_dim=2):
        super().__init__()
        self.linear = torch.nn.Linear(in_dim, out_dim)

class SimpleMLXModel:
    """
    A minimal MLX-like model that stores parameters in a dict.
    For real usage, you'd use actual MLX modules rather than this toy example.
    """
    def __init__(self, in_dim=3, out_dim=2):
        # Using a dictionary to simulate trainable parameters
        # in a shape that matches the Torch linear layer above
        weight = np.random.randn(out_dim, in_dim).astype(np.float32)
        bias   = np.random.randn(out_dim).astype(np.float32)

        # If you have MLX installed, store them as mx.array
        if HAS_MLX:
            self.params = {
                "linear.weight": mx.array(weight),
                "linear.bias":   mx.array(bias),
            }
        else:
            self.params = {
                "linear.weight": weight,
                "linear.bias":   bias,
            }

    def parameters(self):
        """Return a dict (or nested structure) of all parameters."""
        return self.params

# ---------------------------------------------------------------------
# 2) Define fixtures to create fresh models for each test
# ---------------------------------------------------------------------

@pytest.fixture
def torch_modelA():
    """Returns a new instance of SimpleTorchModel."""
    model = SimpleTorchModel()
    return model

@pytest.fixture
def torch_modelB():
    """Another instance of SimpleTorchModel."""
    model = SimpleTorchModel()
    return model

@pytest.fixture
def mlx_modelA():
    """Returns a new instance of SimpleMLXModel, if MLX is available."""
    if not HAS_MLX:
        pytest.skip("MLX not installed in environment.")
    return SimpleMLXModel()

@pytest.fixture
def mlx_modelB():
    """Another instance of SimpleMLXModel."""
    if not HAS_MLX:
        pytest.skip("MLX not installed in environment.")
    return SimpleMLXModel()

# ---------------------------------------------------------------------
# 3) Torch -> Torch Tests
# ---------------------------------------------------------------------

def test_torch_to_torch_strict(torch_modelA, torch_modelB):
    """
    Test copying all params from torch_modelA to torch_modelB in strict mode.
    They have identical architectures, so no KeyErrors or shape mismatches expected.
    """
    # Initialize modelA with known random weights
    with torch.no_grad():
        torch_modelA.linear.weight.fill_(1.23)
        torch_modelA.linear.bias.fill_(-0.99)

    # Copy
    copy_weights(torch_modelA, torch_modelB, strict=True)

    # Check they match
    assert torch.allclose(torch_modelB.linear.weight, torch_modelA.linear.weight)
    assert torch.allclose(torch_modelB.linear.bias, torch_modelA.linear.bias)

def test_torch_to_torch_nonstrict_missing_key(torch_modelA):
    """
    If we artificially remove a parameter from modelB's state_dict,
    in non-strict mode, we should skip copying that key (with a warning).
    """
    # Create a modelB with an extra param name or remove something
    class ModifiedTorchModel(SimpleTorchModel):
        def __init__(self):
            super().__init__()
            # rename the layer
            self.dummy_linear = self.linear
            del self.linear

    modelB = ModifiedTorchModel()

    # This should NOT raise an error in non-strict mode,
    # but it will log a warning about missing keys.
    copy_weights(torch_modelA, modelB, strict=False)

# ---------------------------------------------------------------------
# 4) MLX -> MLX Tests
# ---------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MLX, reason="MLX is not installed.")
def test_mlx_to_mlx_strict(mlx_modelA, mlx_modelB):
    """
    Test copying from one MLX model to another in strict mode.
    Should pass if they have matching parameter names + shapes.
    """
    # Modify modelA so we know the "source" values
    w = np.ones((2, 3), dtype=np.float32) * 4.2
    b = np.ones((2,), dtype=np.float32) * -1.1

    mlx_modelA.params["linear.weight"] = mx.array(w)
    mlx_modelA.params["linear.bias"]   = mx.array(b)

    # Copy
    copy_weights(mlx_modelA, mlx_modelB, strict=True)

    # Check if modelB received them
    assert np.allclose(mlx_modelB.params["linear.weight"].asnumpy(), w)
    assert np.allclose(mlx_modelB.params["linear.bias"].asnumpy(), b)

@pytest.mark.skipif(not HAS_MLX, reason="MLX is not installed.")
def test_mlx_to_mlx_missing_key_nonstrict(mlx_modelA):
    """
    If the destination is missing one param, non-strict should skip it with a warning.
    """
    class MissingParamMLXModel:
        def __init__(self):
            if HAS_MLX:
                # Let's only store the 'linear.bias' param
                self.params = {
                    "linear.bias": mx.array(np.zeros((2,), dtype=np.float32))
                }
            else:
                self.params = {"linear.bias": np.zeros((2,), dtype=np.float32)}

        def parameters(self):
            return self.params

    modelB = MissingParamMLXModel()
    copy_weights(mlx_modelA, modelB, strict=False)
    # Should not raise KeyError.

# ---------------------------------------------------------------------
# 5) Cross-Framework Tests
# ---------------------------------------------------------------------

@pytest.mark.skipif(not HAS_MLX, reason="MLX is not installed.")
def test_torch_to_mlx_strict(torch_modelA, mlx_modelB):
    """
    Copy from a simple Torch model to MLX model, strict mode.
    They have matching param names & shapes.
    """
    with torch.no_grad():
        torch_modelA.linear.weight.fill_(2.0)
        torch_modelA.linear.bias.fill_(-3.0)

    copy_weights(torch_modelA, mlx_modelB, strict=True)

    # Check MLX modelB
    w_mlx = mlx_modelB.params["linear.weight"].asnumpy()
    b_mlx = mlx_modelB.params["linear.bias"].asnumpy()

    expected_w = torch_modelA.linear.weight.detach().cpu().numpy()
    expected_b = torch_modelA.linear.bias.detach().cpu().numpy()

    assert np.allclose(w_mlx, expected_w)
    assert np.allclose(b_mlx, expected_b)

@pytest.mark.skipif(not HAS_MLX, reason="MLX is not installed.")
def test_mlx_to_torch_strict(mlx_modelA, torch_modelB):
    """
    Copy from MLX model to Torch model, strict mode.
    They have matching param names & shapes.
    """
    # Set MLX model param data
    w = np.random.randn(2, 3).astype(np.float32)
    b = np.random.randn(2).astype(np.float32)
    mlx_modelA.params["linear.weight"] = mx.array(w)
    mlx_modelA.params["linear.bias"]   = mx.array(b)

    copy_weights(mlx_modelA, torch_modelB, strict=True)

    # Check Torch modelB
    w_torch = torch_modelB.linear.weight.data
    b_torch = torch_modelB.linear.bias.data

    assert np.allclose(w_torch.numpy(), w)
    assert np.allclose(b_torch.numpy(), b)

# ---------------------------------------------------------------------
# 6) Shape Mismatch Tests
# ---------------------------------------------------------------------

def test_torch_to_torch_strict_shape_mismatch():
    """
    Torch->Torch: If shape mismatch occurs in strict mode, expect a RuntimeError or ValueError.
    We'll artificially change the shape in destination.
    """
    src = SimpleTorchModel(in_dim=3, out_dim=2)
    dst = SimpleTorchModel(in_dim=4, out_dim=2)  # mismatch in the weight shape

    with pytest.raises(RuntimeError):
        copy_weights(src, dst, strict=True)

@pytest.mark.skipif(not HAS_MLX, reason="MLX is not installed.")
def test_torch_to_mlx_strict_shape_mismatch(torch_modelA, mlx_modelB):
    """
    Torch->MLX: If shapes mismatch and we are in strict mode, expect a ValueError.
    We'll artificially alter the MLX param shape.
    """
    # Overwrite modelB to have different shape, e.g. out_dim=3, in_dim=3
    wrong_shape = np.zeros((3, 3), dtype=np.float32)
    mlx_modelB.params["linear.weight"] = mx.array(wrong_shape)  # mismatch

    with pytest.raises(ValueError):
        copy_weights(torch_modelA, mlx_modelB, strict=True)
