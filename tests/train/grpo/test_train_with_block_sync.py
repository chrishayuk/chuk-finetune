# tests/train/grpo/test_train_with_block_sync.py

import pytest
from unittest.mock import patch, MagicMock

from train.grpo.train_with_block_sync import train_grpo_block_sync


@pytest.mark.parametrize("use_checkpoints", [False, True])
def test_train_grpo_block_sync(use_checkpoints):
    """
    Tests the train_grpo_block_sync function under two scenarios:
      1) No checkpointing
      2) Checkpointing every 2 batches and every 1 epoch
    Ensures block-sync triggers after 'sync_every_n_batches', and verifies final returns.
    """

    # 1) Mock out get_dataloader so each epoch yields 2 batches
    #    We'll return a function that, when called, yields 2 "batches".
    def mock_data_iter_fn():
        yield ["batch_item_1"]
        yield ["batch_item_2"]

    # We'll have 'get_dataloader' return the above *function* each time it's called.
    mock_get_dataloader = MagicMock(return_value=mock_data_iter_fn)

    # 2) Mock out train_grpo: each call returns a different (loss, reward)
    #    so we can confirm the final returned (loss, reward) are from the last call.
    train_grpo_call_results = [
        (0.1, 1.1),
        (0.2, 1.2),
        (0.3, 1.3),
        (0.4, 1.4),
    ]
    # We'll pop results from the front each time train_grpo is called
    def mock_train_grpo(*args, **kwargs):
        return train_grpo_call_results.pop(0)

    # 3) Mocks for copy_weights & is_torch_model
    mock_copy_weights = MagicMock()
    mock_is_torch_model = MagicMock(return_value=True)

    # 4) Mock for save_checkpoint
    mock_save_checkpoint = MagicMock()

    # We'll run 2 epochs, each with 2 batches => total 4 calls to train_grpo.
    # We'll set sync_every_n_batches=2 => triggers a sync after the 2nd batch each epoch.
    # We'll also optionally set checkpointing parameters if use_checkpoints=True.

    checkpoint_dir = "checkpoints_dir" if use_checkpoints else None
    checkpoint_every_n_batches = 2 if use_checkpoints else None
    checkpoint_every_n_epochs = 1 if use_checkpoints else None

    # 5) Patch the dependent functions/classes
    with patch("train.grpo.train_with_block_sync.get_dataloader", mock_get_dataloader), \
         patch("train.grpo.train_with_block_sync.train_grpo", side_effect=mock_train_grpo) as mock_train, \
         patch("train.grpo.train_with_block_sync.copy_weights", mock_copy_weights) as mock_copy, \
         patch("train.grpo.train_with_block_sync.is_torch_model", mock_is_torch_model), \
         patch("train.grpo.train_with_block_sync.save_checkpoint", mock_save_checkpoint):

        # 6) Call the function under test
        final_loss, final_reward = train_grpo_block_sync(
            base_model=MagicMock(name="base_model"),
            ref_model=MagicMock(name="ref_model"),
            tokenizer=MagicMock(name="tokenizer"),
            dataset=["fake_data1", "fake_data2"],  # we only care that get_dataloader is called
            calculate_reward=MagicMock(name="calculate_reward"),
            lr=1e-4,
            total_epochs=2,
            batch_size=4,
            G=2,
            device="cpu",
            verbose=False,
            kl_coeff=0.1,
            sync_every_n_batches=2,
            shuffle=False,
            checkpoint_dir=checkpoint_dir,
            checkpoint_every_n_batches=checkpoint_every_n_batches,
            checkpoint_every_n_epochs=checkpoint_every_n_epochs
        )

    # 7) Assertions

    # a) Check that get_dataloader was called once per epoch => total 2 times
    assert mock_get_dataloader.call_count == 2

    # b) train_grpo was called 4 times (2 batches * 2 epochs)
    assert mock_train.call_count == 4

    # c) The final returned (loss, reward) should be from the last (4th) call => (0.4, 1.4)
    assert final_loss == 0.4
    assert final_reward == 1.4

    # d) check block sync calls => copy_weights is called:
    #    - once at the start of each epoch (2 epochs => 2 times)
    #    - once after each block of 2 batches (2 epochs => each has 2 batches => after each epoch's 2nd batch => 2 times)
    # => total 4 calls
    # but you can check each call or just the total:
    assert mock_copy_weights.call_count == 4

    # e) if is_torch_model(ref_model) => ref_model.eval() is called
    #    We didn't explicitly track ref_model, but we can see if we want to check .eval()
    #    If you did: ref_model = MagicMock(name="ref_model"), you can check ref_model.eval.call_count.

    # f) Checkpoint calls
    if use_checkpoints:
        # With checkpoint_every_n_batches=2 and 2 total batches per epoch, we checkpoint after
        # each epoch's second batch => so 2 times from batch-checkpoints
        # plus checkpoint_every_n_epochs=1 => checkpoint after each epoch => 2 times
        # total 4 calls to save_checkpoint
        assert mock_save_checkpoint.call_count == 4
    else:
        # No checkpointing => no calls
        mock_save_checkpoint.assert_not_called()
