# tests/train/torch/test_grpo_generation.py

import pytest
import torch
from unittest.mock import patch, MagicMock

from train.grpo.torch.grpo_generation import generate_single_response_and_oldlogprob

def test_generate_single_response_normal():
    """
    Normal path:
      - top_p_generate_torch_with_kvcache => non-empty text
      - tokenizer => non-empty input_ids
      - gather_logprobs => returns a float
    """
    mock_ref_model = MagicMock(name="ref_model")
    # Use a real torch.device => CPU
    mock_ref_model.device = torch.device("cpu")

    mock_tokenizer = MagicMock(name="tokenizer")

    with patch("train.grpo.torch.grpo_generation.top_p_generate_torch_with_kvcache",
               return_value=" Hello world") as mock_gen:
        # tokenizer(...) => shape [1,3]
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[10, 11, 12]])
        }
        mock_tokenizer.eos_token_id = 2

        with patch("train.grpo.torch.grpo_generation.gather_logprobs",
                   return_value=torch.tensor([7.5])) as mock_gather:
            mock_logits = torch.randn(1, 3, 50)
            mock_output = MagicMock(name="model_output", logits=mock_logits)
            # So that calling ref_model(...) returns mock_output
            mock_ref_model.return_value = mock_output

            resp_text, sum_lp = generate_single_response_and_oldlogprob(
                ref_model=mock_ref_model,
                tokenizer=mock_tokenizer,
                prompt="Some prompt"
            )

    assert resp_text == "<think>Hello world"  # ' Hello world'.strip() => 'Hello world'
    assert sum_lp == 7.5

    mock_gen.assert_called_once()
    mock_tokenizer.assert_called_once_with("<think>Hello world", return_tensors="pt")
    mock_gather.assert_called_once()
    # Confirm gather_logprobs got the correct logits & input_ids
    call_args, _ = mock_gather.call_args
    logits_arg, input_ids_arg = call_args
    assert torch.equal(logits_arg, mock_logits)
    assert torch.equal(input_ids_arg, torch.tensor([[10, 11, 12]]))


def test_generate_empty_generation():
    """
    If top_p_generate_torch_with_kvcache returns empty => fallback => <|endoftext|>.
    """
    mock_ref_model = MagicMock(name="ref_model")
    mock_ref_model.device = torch.device("cpu")

    mock_tokenizer = MagicMock(name="tokenizer")
    mock_tokenizer.eos_token_id = 999

    with patch("train.grpo.torch.grpo_generation.top_p_generate_torch_with_kvcache",
               return_value="") as mock_gen, \
         patch("train.grpo.torch.grpo_generation.gather_logprobs",
               return_value=torch.tensor([0.0])) as mock_gather:

        # Suppose tokenizer => shape [1,2]
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[7, 8]])
        }

        mock_logits = torch.randn(1, 2, 50)
        mock_output = MagicMock(name="model_output", logits=mock_logits)
        mock_ref_model.return_value = mock_output

        resp_text, sum_lp = generate_single_response_and_oldlogprob(
            ref_model=mock_ref_model,
            tokenizer=mock_tokenizer,
            prompt="Another prompt"
        )

    assert resp_text == "<think><|endoftext|>"
    assert sum_lp == 0.0
    mock_gen.assert_called_once()
    mock_tokenizer.assert_called_once_with("<think><|endoftext|>", return_tensors="pt")
    mock_gather.assert_called_once()
    call_args, _ = mock_gather.call_args
    logits_arg, input_ids_arg = call_args
    assert torch.equal(logits_arg, mock_logits)
    assert torch.equal(input_ids_arg, torch.tensor([[7, 8]]))


def test_generate_empty_tokenizer():
    """
    If tokenizer(...) => input_ids is empty => fallback => [eos_token_id].
    """
    mock_ref_model = MagicMock(name="ref_model")
    mock_ref_model.device = torch.device("cpu")

    mock_tokenizer = MagicMock(name="tokenizer")
    eos_id = 42
    mock_tokenizer.eos_token_id = eos_id

    with patch("train.grpo.torch.grpo_generation.top_p_generate_torch_with_kvcache",
               return_value="my text") as mock_gen, \
         patch("train.grpo.torch.grpo_generation.gather_logprobs",
               return_value=torch.tensor([-2.5])) as mock_gather:

        # Return shape [1,0] => empty
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([]).view(1, 0)
        }
        mock_logits = torch.randn(1, 1, 50)
        mock_output = MagicMock(name="model_output", logits=mock_logits)
        mock_ref_model.return_value = mock_output

        resp_text, sum_lp = generate_single_response_and_oldlogprob(
            ref_model=mock_ref_model,
            tokenizer=mock_tokenizer,
            prompt="Prompt here"
        )

    assert resp_text == "<think>my text"
    assert sum_lp == -2.5
    mock_gen.assert_called_once()
    mock_tokenizer.assert_called_once_with("<think>my text", return_tensors="pt")
    mock_gather.assert_called_once()
    call_args, _ = mock_gather.call_args
    logits_arg, input_ids_arg = call_args
    # Fallback => shape [1,1], eos token
    assert input_ids_arg.shape == (1, 1)
    assert input_ids_arg[0, 0].item() == eos_id


def test_generate_device_transfer():
    """
    Check that input_ids are moved to the same device as ref_model.
    We'll use CPU so the test doesn't require CUDA.
    """
    mock_ref_model = MagicMock(name="ref_model")
    # We'll say it's on CPU
    mock_ref_model.device = torch.device("cpu")

    mock_tokenizer = MagicMock(name="tokenizer")
    mock_tokenizer.eos_token_id = 3

    with patch("train.grpo.torch.grpo_generation.top_p_generate_torch_with_kvcache",
               return_value=" Hello") as mock_gen, \
         patch("train.grpo.torch.grpo_generation.gather_logprobs",
               return_value=torch.tensor([5.0])) as mock_gather:

        # Return shape [1,2] on CPU
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[100, 101]])  # CPU by default
        }
        mock_logits = torch.randn(1, 2, 50)
        mock_output = MagicMock(name="model_output", logits=mock_logits)
        mock_ref_model.return_value = mock_output

        resp_text, sum_lp = generate_single_response_and_oldlogprob(
            ref_model=mock_ref_model,
            tokenizer=mock_tokenizer,
            prompt="Testing device"
        )

    assert resp_text == "<think>Hello"
    assert sum_lp == 5.0
    mock_gen.assert_called_once()
    mock_tokenizer.assert_called_once_with("<think>Hello", return_tensors="pt")
    mock_gather.assert_called_once()

    # Ensure we moved input_ids => same device as ref_model => CPU
    call_args, _ = mock_gather.call_args
    logits_arg, input_ids_arg = call_args
    assert logits_arg.device == torch.device("cpu")
    assert input_ids_arg.device == torch.device("cpu")
