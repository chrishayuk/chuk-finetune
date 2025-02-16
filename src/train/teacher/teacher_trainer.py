# src/train/teacher/teacher_trainer.py

import logging
from typing import List, Any, Optional

from train.trainer_base import Trainer
from train.teacher.teacher_data_prepare import prepare_batch_data_for_teacher

logger = logging.getLogger(__name__)

class TeacherTrainer(Trainer):
    """
    A specialized trainer that collects data from a 'teacher' model without performing
    any parameter updates (e.g., no PPO/GRPO).
    
    Key points:
      - 'self.model' is a teacher model (Torch or MLX) used strictly for data generation.
      - We generate responses for each batch item, calculate rewards, and skip items that
        produce a None reward or fail per-item min_reward (if defined).
      - We do not optimize or fine-tune the teacher model in this trainer.

    Workflow:
      - 'prepare_batch_data(...)' calls 'prepare_batch_data_for_teacher(...)'
        to generate teacher responses, evaluate rewards, and skip items as needed.
      - 'train_step(...)' simply computes and returns the mean reward (for logging),
        along with the filtered batch data (no weight updates).
    """

    def __init__(
        self,
        teacher_model,
        tokenizer,
        calculate_reward,
        generate_single_fn,
        G: int = 4,
        device=None,
        verbose: bool = False,
        prompt_template: Optional[str] = None,
        default_min_reward: float = 1.0
    ):
        """
        Initialize the TeacherTrainer.

        Args:
            teacher_model: The teacher model (Torch or MLX) used for text generation. 
                           This model remains frozen or in eval mode.
            tokenizer: The associated tokenizer for the teacher model.
            calculate_reward: A function (resp_text, item_dict) -> (score, feedback).
                              If 'score' is None, the item is skipped entirely.
            generate_single_fn: A function that generates a single teacher response 
                                given (model, tokenizer, prompt, verbose, ...).
                                This function is wrapped internally to automatically
                                pass 'self.model' and 'self.tokenizer'.
            G (int): Number of responses to generate per item in the batch. Default is 4.
            device: The device to run the teacher model on, e.g. 'cpu', 'cuda', or 'mlx'.
                    Handled internally by the trainer base.
            verbose: If True, logs debug info about generation steps, skipping items, etc.
            prompt_template (str, optional): A template string that includes '{{question}}'
                                             as a placeholder for the original prompt. 
                                             If provided, each prompt is transformed.
            default_min_reward (float): If an item does not specify 'min_reward', 
                                        this is used as the fallback threshold. 
                                        Defaults to 1.0.
        """
        # No optimizer is passed because we do not update model parameters in this trainer.
        super().__init__(
            teacher_model,
            tokenizer,
            optimizer=None,
            device=device,
            verbose=verbose
        )

        # Setup user-provided functions & settings
        self.calculate_reward = calculate_reward
        self.generate_single_fn = generate_single_fn
        self.G = G
        self.prompt_template = prompt_template
        self.default_min_reward = default_min_reward

    def prepare_batch_data(self, batch_questions: List[Any]):
        """
        Prepare the batch data for teacher data collection.

        Steps:
          1) Wraps 'generate_single_fn' so it automatically includes 'self.model' 
             and 'self.tokenizer'.
          2) Calls 'prepare_batch_data_for_teacher' to generate G responses, 
             calculate rewards, and discard items with None reward.
          3) Further filters out any item that does not meet *its own* min_reward, 
             if provided, or 'default_min_reward' otherwise.

        Each item can have 'min_reward': 
          - If so, at least one response must be >= that threshold to keep the item.
          - If not present, we use 'self.default_min_reward'.

        Args:
            batch_questions: A list of input items (dicts or other structures) each 
                             typically containing:
                               "prompt": str,
                               "min_reward": float (optional),
                               ...
        Returns:
            A list of processed batch items. Each item includes:
              {
                "item": original (or template-transformed) input item,
                "responses": [list of generated responses],
                "teacher_logprobs": [list of log-prob or likelihood values],
                "rewards": [list of reward scores],
                "feedbacks": [list of verifier feedback strings (if included)]
              }

        Skips items entirely if:
          - any generated response yields a None reward, OR
          - none of the responses meet the (item-specific or default) min_reward threshold.
        """
        # 1) Wrap the generation function
        wrapped_generate_fn = lambda prompt, vb: self.generate_single_fn(
            self.model, self.tokenizer, prompt, vb
        )

        # 2) Generate responses
        raw_batch_data = prepare_batch_data_for_teacher(
            batch_questions=batch_questions,
            generate_single_fn=wrapped_generate_fn,
            calculate_reward=self.calculate_reward,
            G=self.G,
            verbose=self.verbose,
            prompt_template=self.prompt_template 
        )

        # 3) Filter by per-item min_reward
        filtered_batch = []
        for entry in raw_batch_data:
            # The item dict that includes 'prompt', optional 'min_reward', etc.
            item_dict = entry["item"]

            # Grab the item-specific min_reward or fallback
            item_min_reward = item_dict.get("min_reward", self.default_min_reward)
            rewards = entry["rewards"]

            # Check if at least one response meets item_min_reward
            if any(r >= item_min_reward for r in rewards):
                filtered_batch.append(entry)
            else:
                if self.verbose:
                    logger.info(
                        f"[SKIP] Item with prompt='{item_dict.get('prompt', '')[:60]}...' "
                        f"did not meet min_reward={item_min_reward}, rewards={rewards}"
                    )

        return filtered_batch

    def train_step(self, batch_data):
        """
        Compute the average reward and return the batch data, without model updates.

        If 'batch_data' is empty, returns (0.0, batch_data).

        Args:
            batch_data: A list of processed (and filtered) items.

        Returns:
            (mean_reward, batch_data)
        """
        if not batch_data:
            return 0.0, batch_data

        all_rewards = []
        for item in batch_data:
            all_rewards.extend(item["rewards"])

        mean_reward = sum(all_rewards) / len(all_rewards) if all_rewards else 0.0
        return mean_reward, batch_data