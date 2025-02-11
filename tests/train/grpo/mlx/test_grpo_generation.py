# tests/train/grpo/mlx/test_grpo_generation.py

import pytest
from unittest.mock import patch, MagicMock, ANY

from train.grpo.mlx.grpo_generation import generate_single_response_and_oldlogprob

def test_generate_single_response_normal():
    """
    Test the typical path:
      - top_p_generate returns a nonempty string
      - tokenize => nonempty tokens
      - ref_model(...) => some dummy logits
      - gather_logprobs => sum of log-probs
    """
    mock_ref_model = MagicMock(name="ref_model")
    mock_tokenizer = MagicMock(name="tokenizer")
    # Suppose eos_token_id = 1
    mock_tokenizer.eos_token_id = 1

    # 1) top_p_generate => returns " Hello world"
    with patch(
        "train.grpo.mlx.grpo_generation.top_p_generate",
        return_value=" Hello world"
    ) as mock_top_p_generate:

        # 2) tokenizer.encode => returns some tokens e.g. [3, 4, 5]
        mock_tokenizer.encode.return_value = [3, 4, 5]

        # 3) We'll patch gather_logprobs for a unit-level test
        with patch(
            "train.grpo.mlx.grpo_generation.gather_logprobs",
            return_value=7.5
        ) as mock_gather:

            # Also mock mx.array so we don't run real MLX code
            with patch("train.grpo.mlx.grpo_generation.mx.array") as mock_mx_array:
                mock_mx_array.return_value = MagicMock(name="mx_array")
                
                # ref_model(...) => returns dummy logits => shape [1, seq_len, vocab_size]
                dummy_logits = MagicMock(name="dummy_logits")
                mock_ref_model.return_value = dummy_logits

                # Call function
                resp_text, sum_lp = generate_single_response_and_oldlogprob(
                    ref_model=mock_ref_model,
                    tokenizer=mock_tokenizer,
                    prompt="Some prompt"
                )

    # Assertions
    # top_p_generate called once with correct arguments
    mock_top_p_generate.assert_called_once()

    # Because raw_resp.strip() => "Hello world",
    # and code does "<think>" + "Hello world" => "<think>Hello world"
    assert resp_text == "<think>Hello world"
    assert sum_lp == 7.5

    # Check that encode was called with the final text
    mock_tokenizer.encode.assert_called_once_with("<think>Hello world")

    # ref_model(...) => called once with mx.array
    mock_ref_model.assert_called_once()
    mock_mx_array.assert_called_once_with([3, 4, 5], ANY)

    # gather_logprobs called once with (dummy_logits, [3,4,5])
    mock_gather.assert_called_once_with(dummy_logits, [3, 4, 5])


def test_generate_single_response_empty_generation():
    """
    If top_p_generate returns an empty string => fallback => <|endoftext|>.
    """
    mock_ref_model = MagicMock(name="ref_model")
    mock_tokenizer = MagicMock(name="tokenizer")
    mock_tokenizer.eos_token_id = 1

    with patch("train.grpo.mlx.grpo_generation.top_p_generate", return_value="") as mock_top_p_generate, \
         patch("train.grpo.mlx.grpo_generation.gather_logprobs", return_value=0.0) as mock_gather, \
         patch("train.grpo.mlx.grpo_generation.mx.array") as mock_mx_array:

        # We'll simulate that tokenizer.encode => e.g. [9], for the final text "<think><|endoftext|>"
        mock_tokenizer.encode.return_value = [9]
        dummy_logits = MagicMock(name="dummy_logits")
        mock_ref_model.return_value = dummy_logits

        resp_text, sum_lp = generate_single_response_and_oldlogprob(
            ref_model=mock_ref_model,
            tokenizer=mock_tokenizer,
            prompt="Some prompt"
        )

    assert resp_text == "<think><|endoftext|>"
    assert sum_lp == 0.0

    mock_top_p_generate.assert_called_once()
    mock_tokenizer.encode.assert_called_once_with("<think><|endoftext|>")
    mock_ref_model.assert_called_once()
    mock_gather.assert_called_once_with(dummy_logits, [9])


def test_generate_single_response_empty_tokens():
    """
    If tokenizer.encode returns an empty list => fallback => [eos_token_id].
    """
    mock_ref_model = MagicMock(name="ref_model")
    mock_tokenizer = MagicMock(name="tokenizer")
    mock_tokenizer.eos_token_id = 99

    with patch("train.grpo.mlx.grpo_generation.top_p_generate", return_value="my text") as mock_top_p_generate, \
         patch("train.grpo.mlx.grpo_generation.gather_logprobs", return_value=-3.7) as mock_gather, \
         patch("train.grpo.mlx.grpo_generation.mx.array") as mock_mx_array:

        # If encode returns empty => fallback to [99]
        mock_tokenizer.encode.return_value = []
        dummy_logits = MagicMock(name="dummy_logits")
        mock_ref_model.return_value = dummy_logits

        resp_text, sum_lp = generate_single_response_and_oldlogprob(
            ref_model=mock_ref_model,
            tokenizer=mock_tokenizer,
            prompt="Another prompt"
        )

    assert resp_text == "<think>my text"
    assert sum_lp == -3.7

    mock_top_p_generate.assert_called_once()
    mock_tokenizer.encode.assert_called_once_with("<think>my text")
    mock_mx_array.assert_called_once_with([99], ANY)  # fallback to eos
    mock_gather.assert_called_once_with(dummy_logits, [99])
