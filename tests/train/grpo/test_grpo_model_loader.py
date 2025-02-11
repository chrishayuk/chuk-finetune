# tests/train/grpo/test_grpo_model_loader.py
import pytest
from unittest.mock import patch, MagicMock

# The function we are testing:
from train.grpo.grpo_model_loader import load_models

@pytest.mark.parametrize(
    "is_mlx_base, is_mlx_ref, device_override, expected_device",
    [
        # 1) Both models are MLX => final_device="mlx"
        (True,  True,  "cuda", "mlx"),
        # 2) Only base model is MLX => final_device="mlx"
        (True,  False, "cuda", "mlx"),
        # 3) Only ref model is MLX => final_device="mlx"
        (False, True,  "cuda", "mlx"),
        # 4) Neither is MLX => final_device=device_override
        (False, False, "cuda", "cuda"),
        # 5) Neither is MLX, device_override=None => "cpu"
        (False, False, None,   "cpu"),
    ]
)
def test_load_models(is_mlx_base, is_mlx_ref, device_override, expected_device):
    mock_base_model = MagicMock(name="base_model")
    mock_tokenizer  = MagicMock(name="tokenizer")
    mock_ref_model  = MagicMock(name="ref_model")

    def mock_loader_side_effect(*args, **kwargs):
        if not hasattr(mock_loader_side_effect, "called_once"):
            mock_loader_side_effect.called_once = True
            return (mock_base_model, mock_tokenizer, is_mlx_base)
        else:
            return (mock_ref_model, None, is_mlx_ref)

    patch_path = "train.grpo.grpo_model_loader.load_model_and_tokenizer"

    with patch(patch_path, side_effect=mock_loader_side_effect) as mock_loader:
        base_model, ref_model, tokenizer, final_device = load_models(
            model_name="dummy-model",
            device_override=device_override
        )

        # Confirm calls
        assert mock_loader.call_count == 2, "Should load base + ref model"

        assert base_model is mock_base_model
        assert ref_model is mock_ref_model
        assert tokenizer is mock_tokenizer
        assert final_device == expected_device

        if is_mlx_ref:
            ref_model.freeze.assert_called_once()
            ref_model.eval.assert_not_called()
        else:
            ref_model.eval.assert_called_once()
            ref_model.freeze.assert_not_called()
