# src/train/teacher/run_teacher_collection.py
import logging
from typing import Optional, Callable, Any, List

# imports
from train.dataset_loader import get_dataloader
from train.teacher.teacher_trainer import TeacherTrainer

# logging
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
    generate_single_fn: Optional[Callable[[str, bool], Any]] = None
):
    """
    Single-pass teacher data collection. 
    We no longer handle 'ensure_dict_fn' => that logic is inlined in teacher_data_prepare.

    :param teacher_model: teacher model (Torch or MLX).
    :param tokenizer: model tokenizer.
    :param dataset: list of items or a dataset.
    :param calculate_reward: (resp_text, item) => (score, feedback). if score=None => skip item.
    :param batch_size: int, how many items per mini-batch
    :param G: number of responses per item
    :param device: e.g. 'cpu','cuda','mlx' => we map to 'torch' or 'mlx'
    :param verbose: debug logs
    :param generate_single_fn: if None, define a default that appends ' => teacher_out_def' + logprob=1.23
    """
    if generate_single_fn is None:
        def default_gen_fn(prompt, vb=False):
            resp = prompt + " => teacher_out_def"
            return (resp, 1.23)
        generate_single_fn = default_gen_fn

    # Build the trainer
    trainer = TeacherTrainer(
        teacher_model=teacher_model,
        tokenizer=tokenizer,
        calculate_reward=calculate_reward,
        generate_single_fn=generate_single_fn,
        G=G,
        device=device,
        verbose=verbose
    )

    # Map device => framework
    framework = "mlx" if device == "mlx" else "torch"
    data_iter_fn = get_dataloader(framework, dataset, batch_size, shuffle=False)

    final_data = []
    logger.info("Starting single-pass teacher data collection...")

    for batch_items in data_iter_fn():
        # prepare the batch data
        batch_data = trainer.prepare_batch_data(batch_items)

        # perform the training step
        mean_reward, result_data = trainer.train_step(batch_data)

        # get the result
        final_data.extend(result_data)

    logger.info(
        f"Collected {len(final_data)} items in single pass over dataset."
    )
    return final_data