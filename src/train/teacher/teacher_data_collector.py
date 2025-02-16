# src/train/teacher/teacher_data_collection.py
import logging
from typing import Optional, Callable, Any, List

# imports
from train.dataset_loader import get_dataloader
from train.teacher.teacher_trainer import TeacherTrainer

logger = logging.getLogger(__name__)

def collect_teacher_data_once(
    teacher_model,
    tokenizer,
    dataset,
    calculate_reward,
    batch_size=4,
    G=4,
    device=None,
    verbose=False,
    generate_single_fn: Optional[Callable[[str, bool], Any]] = None,
    prompt_template: Optional[str] = None,
) -> List[dict]:
    """
    Single-pass teacher data collection, optionally applying a prompt template.

    NOTE: We removed global min_reward / best-response filtering. Instead, each
          dataset item may specify its own 'min_reward' in the JSON. If none
          of a given item's responses meet that threshold, we skip the item
          during data preparation (see 'prepare_batch_data_for_teacher').

    Args:
        teacher_model: The teacher model (Torch or MLX).
        tokenizer: The associated tokenizer.
        dataset: A list of items or a dataset object.
        calculate_reward: Function (resp_text, item) -> (score, feedback).
                          If 'score' is None, the entire item is skipped.
        batch_size: How many items per mini-batch (int).
        G: Number of responses per item (int).
        device: 'cpu', 'cuda', 'mlx', etc., to specify the device.
        verbose: Whether to log debug messages (bool).
        generate_single_fn: A custom generation function with signature
            (prompt: str, verbose: bool) -> (response_text: str, log_prob: float).
            If None, a default stub is used.
        prompt_template: An optional template string with '{{question}}' for the original prompt.

    Returns:
        A list of data dicts. Each item has the final 'prompt' (possibly templated),
        'responses', 'teacher_logprobs', 'rewards', and 'feedbacks'.
        Items whose responses do not meet their per-item min_reward (if defined)
        are skipped automatically in 'prepare_batch_data_for_teacher'.
    """
    # Build the trainer, passing in the optional prompt_template
    trainer = TeacherTrainer(
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        calculate_reward=calculate_reward,
        generate_single_fn=generate_single_fn,
        G=G,
        device=device,
        verbose=verbose,
        prompt_template=prompt_template
    )

    # Map device => framework
    framework = "mlx" if device == "mlx" else "torch"
    data_iter_fn = get_dataloader(framework, dataset, batch_size, shuffle=False)

    final_data = []
    logger.info("Starting single-pass teacher data collection...")

    for batch_items in data_iter_fn():
        # Prepare the batch data (including template injection if provided).
        # This step also applies per-item min_reward logic if present.
        batch_data = trainer.prepare_batch_data(batch_items)

        # Perform the 'train' step (in reality, just calculating mean rewards; no updates).
        mean_reward, result_data = trainer.train_step(batch_data)

        # Accumulate the results
        final_data.extend(result_data)

    # log it
    logger.info(f"Collected {len(final_data)} items in single pass over dataset.")

    # return the data
    return final_data