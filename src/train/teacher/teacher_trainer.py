# src/train/teacher/teacher_trainer.py

import logging
from typing import List, Any

from train.trainer_base import Trainer
from train.teacher.teacher_data_prepare import prepare_batch_data_for_teacher

logger = logging.getLogger(__name__)

class TeacherTrainer(Trainer):
    """
    A simple trainer that collects teacher data but does NOT do PPO/GRPO updates.
    'self.model' is the teacher model (Torch or MLX).

    We define:
      - 'prepare_batch_data(...)' calls 'prepare_batch_data_for_teacher(...)'
        which already handles skipping invalid items or those with reward=None.
      - 'train_step(...)' that computes a mean reward for logging and returns the data.
    """

    def __init__(
        self,
        teacher_model,
        tokenizer,
        calculate_reward,
        generate_single_fn,
        G: int = 4,
        device=None,
        verbose: bool = False
    ):
        """
        :param teacher_model: The teacher (Torch or MLX) used for text generation.
        :param tokenizer: The associated tokenizer for teacher_model.
        :param calculate_reward: A function (resp_text, item_dict) => (score, feedback),
                                 returning None to indicate the item should be skipped.
        :param generate_single_fn: A unified teacher generation function with signature:
                                    generate_single_teacher_response(teacher_model, tokenizer, prompt, verbose, ...)
                                    This function will be wrapped internally.
        :param G: Number of responses to generate per item.
        :param device: e.g. 'cpu', 'cuda', 'mlx' (your code should interpret device => framework).
        :param verbose: If True, logs debug info about generation and skipping.
        """
        # Pass None for optimizer because no training updates occur
        super().__init__(teacher_model, tokenizer, optimizer=None, device=device, verbose=verbose)
        self.calculate_reward = calculate_reward
        self.generate_single_fn = generate_single_fn
        self.G = G

    def prepare_batch_data(self, batch_questions: List[Any]):
        """
        Prepares batch data by calling prepare_batch_data_for_teacher() with a wrapped
        generation function. The wrapper automatically passes self.model and self.tokenizer
        to the unified teacher generation function.
        """
        wrapped_generate_fn = lambda prompt, verbose: self.generate_single_fn(
            self.model, self.tokenizer, prompt, verbose
        )
        batch_data = prepare_batch_data_for_teacher(
            batch_questions=batch_questions,
            generate_single_fn=wrapped_generate_fn,
            calculate_reward=self.calculate_reward,
            G=self.G,
            verbose=self.verbose
        )
        return batch_data

    def train_step(self, batch_data):
        """
        No PPO/GRPO updates are performed. Instead, compute the mean reward from the batch
        for logging purposes, and return the batch data.
        """
        if not batch_data:
            return 0.0, batch_data

        all_rewards = []
        for item in batch_data:
            all_rewards.extend(item["rewards"])

        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        return mean_reward, batch_data
