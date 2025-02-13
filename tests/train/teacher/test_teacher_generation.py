import pytest
import logging
from unittest.mock import MagicMock
import torch

# Import the unified teacher generation function and is_mlx_model checker
from train.teacher.teacher_generation import generate_single_teacher_response, is_mlx_model

logger = logging.getLogger(__name__)

# ------------------ Torch Mocks -------------------

class FakeTorchTokenizer:
    eos_token_id = 999
    def __call__(self, text, return_tensors=None):
        if return_tensors == "pt":
            # Return a real tensor of shape [1,3]
            return {"input_ids": torch.tensor([[101, 102, 103]])}
        return {}
    def encode(self, text):
        # Not used in Torch branch, but define for completeness
        return [101, 102, 103]

@pytest.fixture
def mock_torch_model():
    """
    A minimal Torch-like model mock:
      - .device is set to a real torch.device.
      - The forward pass returns a dummy output with .logits set to a real tensor.
    """
    model = MagicMock()
    model.device = torch.device("cpu")
    
    def mock_forward(**kwargs):
        # Return a dummy logits tensor with shape [1, 3, 10]
        batch_size, seq_len, vocab_size = 1, 3, 10
        logits = torch.zeros((batch_size, seq_len, vocab_size))
        mock_output = MagicMock()
        mock_output.logits = logits
        return mock_output
    model.__call__ = mock_forward
    return model

@pytest.fixture
def mock_torch_tokenizer():
    """
    A minimal Torch-like tokenizer that returns a real dictionary with "input_ids" as a torch.Tensor.
    """
    return FakeTorchTokenizer()

# ------------------ MLX Mocks ---------------------

@pytest.fixture
def mock_mlx_model():
    """
    A minimal MLX-like model mock.
    Has a .freeze() method so that is_mlx_model returns True.
    The forward pass returns a dummy placeholder (its numeric output is patched).
    """
    model = MagicMock()
    model.freeze = MagicMock()
    def mock_forward_mlx(*args, **kwargs):
        return "mlx_logits_placeholder"
    model.__call__ = mock_forward_mlx
    return model

@pytest.fixture
def mock_mlx_tokenizer():
    """
    A minimal MLX-like tokenizer mock.
    Its encode() method returns a simple list of token IDs.
    """
    tokenizer = MagicMock()
    def mock_encode(text):
        return [101, 102, 103]
    tokenizer.encode = mock_encode
    tokenizer.eos_token_id = 999
    return tokenizer

# ------------------ Patch Top-p Generation ------------------

@pytest.fixture
def patch_torch_generation(monkeypatch):
    def mock_torch_generate(*args, **kwargs):
        return "torch_generated_text"
    monkeypatch.setattr(
        "train.teacher.teacher_generation.top_p_generate_torch_with_kvcache",
        mock_torch_generate
    )

@pytest.fixture
def patch_mlx_generation(monkeypatch):
    def mock_mlx_generate(*args, **kwargs):
        return "mlx_generated_text"
    monkeypatch.setattr(
        "train.teacher.teacher_generation.top_p_generate",
        mock_mlx_generate
    )

# ------------------ Patch gather_logprobs ------------------

@pytest.fixture
def patch_gather_logprobs_torch(monkeypatch):
    """
    Patch the local alias 'gather_logprobs_torch' in teacher_generation.py.
    It returns a MagicMock whose .item() method returns 0.123.
    """
    def mock_gather_torch(logits, input_ids):
        mock_tensor = MagicMock()
        mock_tensor.item.return_value = 0.123
        return mock_tensor
    monkeypatch.setattr(
        "train.teacher.teacher_generation.gather_logprobs_torch",
        mock_gather_torch
    )

@pytest.fixture
def patch_gather_logprobs_mlx(monkeypatch):
    """
    Patch the local alias 'gather_logprobs_mlx' in teacher_generation.py to return 0.456.
    """
    def mock_gather_mlx(logits, tokens):
        return 0.456
    monkeypatch.setattr(
        "train.teacher.teacher_generation.gather_logprobs_mlx",
        mock_gather_mlx
    )

# ------------------ Tests ------------------

def test_teacher_generation_torch(
    mock_torch_model, 
    mock_torch_tokenizer,
    patch_torch_generation, 
    patch_gather_logprobs_torch,
    monkeypatch
):
    """
    Test the Torch branch:
    - Force is_mlx_model to return False.
    - Expect top-p generation to return "torch_generated_text".
    - Expect patched gather_logprobs_torch to return 0.123.
    """
    monkeypatch.setattr(
        "train.teacher.teacher_generation.is_mlx_model",
        lambda model: False
    )

    response_text, sum_lp = generate_single_teacher_response(
        teacher_model=mock_torch_model,
        tokenizer=mock_torch_tokenizer,
        prompt="Torch prompt",
        verbose=True,
        max_new_tokens=50,
        temperature=0.8,
        top_p=0.9
    )
    assert response_text == "torch_generated_text"
    assert sum_lp == 0.123

def test_teacher_generation_mlx(
    mock_mlx_model, 
    mock_mlx_tokenizer,
    patch_mlx_generation, 
    patch_gather_logprobs_mlx,
    monkeypatch
):
    """
    Test the MLX branch:
    - Force is_mlx_model to return True.
    - Expect top-p generation to return "mlx_generated_text".
    - Expect patched gather_logprobs_mlx to return 0.456.
    """
    monkeypatch.setattr(
        "train.teacher.teacher_generation.is_mlx_model",
        lambda model: True
    )

    response_text, sum_lp = generate_single_teacher_response(
        teacher_model=mock_mlx_model,
        tokenizer=mock_mlx_tokenizer,
        prompt="MLX prompt",
        verbose=True,
        max_new_tokens=100,
        temperature=0.7,
        top_p=0.95
    )
    assert response_text == "mlx_generated_text"
    assert sum_lp == 0.456
