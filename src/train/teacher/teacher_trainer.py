# src/train/teacher/teacher_trainer.py

import logging
from typing import List, Any

# training imports
from train.trainer_base import Trainer

# teacher imports
from train.teacher.teacher_data_prepare import prepare_batch_data_for_teacher

logger = logging.getLogger(__name__)

class TeacherTrainer(Trainer):
    """
    A simple trainer that collects teacher data but does NOT do PPO/GRPO updates.
    'self.model' is the teacher model (Torch or MLX).

    We define:
      - 'prepare_batch_data(...)' calls 'prepare_batch_data_for_teacher(...)'
        which already handles skipping invalid items or ones with reward=None.
      - 'train_step(...)' that just returns the collected data (no PPO updates),
        computing a mean_reward for logging.
    """

    def __init__(
        self,
        teacher_model,
        tokenizer,
        calculate_reward,
        generate_single_fn,
        G=4,
        device=None,
        verbose=False
    ):
        """
        :param teacher_model: The teacher (Torch or MLX) used for text generation.
        :param tokenizer: The associated tokenizer for teacher_model.
        :param calculate_reward: A function (resp_text, item_dict) => (score, feedback),
                                 returning None => skip item.
        :param generate_single_fn: A function (prompt, verbose=False) => (response_text, logprob).
        :param G: Number of responses to generate per item.
        :param device: e.g. 'cpu', 'cuda', 'mlx' (your code should interpret device => framework).
        :param verbose: if True, logs debug info about generation / skipping.
        """
        # We pass None as 'optimizer', because we are not doing ratio-based updates
        super().__init__(teacher_model, tokenizer, optimizer=None, device=device, verbose=verbose)
        self.calculate_reward = calculate_reward
        self.generate_single_fn = generate_single_fn
        self.G = G

    def prepare_batch_data(self, batch_questions: List[Any]):
        """
        Reuses 'prepare_batch_data_for_teacher' from teacher_data_prepare.py,
        which inlines the check for items that do not have 'prompt', or
        skipping items if reward=None.
        """
        batch_data = prepare_batch_data_for_teacher(
            batch_questions=batch_questions,
            generate_single_fn=self.generate_single_fn,
            calculate_reward=self.calculate_reward,
            G=self.G,
            verbose=self.verbose
        )
        return batch_data

    def train_step(self, batch_data):
        """
        We do NO PPO/GRPO updates. Instead, we just compute a mean reward for logging
        and return the data so a higher-level loop or function can store or process it.
        """
        if not batch_data:
            # If the batch is empty, we can define a "loss" of 0.0 for convenience
            return 0.0, batch_data

        # Flatten out all rewards across the items
        all_rewards = []
        for item in batch_data:
            all_rewards.extend(item["rewards"])

        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        # Return (some_loss, processed_data). We define loss=0.0 since no gradient steps.
        return mean_reward, batch_data
