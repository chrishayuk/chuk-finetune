# src/train/grpo/mlx/distillation_grpo_trainer.py

import ast
import logging
import numpy as np
import mlx.core as mx
import mlx.nn as nn

# Reuse from your code
from train.trainer_base import Trainer
from train.grpo.advantage_utils import compute_advantages
from train.grpo.grpo_prepare import prepare_batch_data_for_grpo
from train.grpo.mlx.grpo_utils import gather_logprobs, gather_kl_divergence
from train.grpo.mlx.grpo_loss import grpo_loss
from train.grpo.mlx.grpo_generation import generate_single_response_and_oldlogprob

logger = logging.getLogger(__name__)

# Color codes for logs (optional)
RESET = "\033[0m"
YELLOW = "\033[93m"

def color_text(text, color):
    return f"{color}{text}{RESET}"

def ensure_dict_mlx(item):
    if isinstance(item, dict):
        return item
    if isinstance(item, str) and item.strip().startswith("{"):
        try:
            possible_dict = ast.literal_eval(item)
            if isinstance(possible_dict, dict):
                return possible_dict
        except:
            pass
    raise ValueError(f"[ERROR] Unexpected non-dict item: {item}")


class DistillationGRPOTrainer(Trainer):
    """
    A trainer that *combines*:
      - GRPO/PPO style updates (using 'ref_model' for KL reference),
      - Distillation from a separate 'teacher_model' that sees a different (long) prompt.

    'self.model' (student) is the *current* model being trained with short prompts.
    'self.ref_model' (frozen) is the old student policy for ratio-based PPO.
    'self.teacher_model' (frozen) sees the *long prompt* for distillation.
    """

    def __init__(
        self,
        model,           # student model (trainable)
        ref_model,       # old snapshot of student model (frozen, for PPO ratio)
        teacher_model,   # separate teacher (frozen), sees long prompt
        tokenizer_short, # tokenizer for short prompt (student)
        tokenizer_long,  # tokenizer for long prompt (teacher)
        optimizer,
        calculate_reward,
        G=4,
        kl_coeff=0.1,
        alpha_distill=0.5,
        device=None,
        verbose=False
    ):
        super().__init__(model, tokenizer_short, optimizer, device=device, verbose=verbose)
        self.ref_model = ref_model
        self.teacher_model = teacher_model
        self.tokenizer_long = tokenizer_long
        self.calculate_reward = calculate_reward
        self.G = G
        self.kl_coeff = kl_coeff
        self.alpha_distill = alpha_distill

    def prepare_batch_data(self, batch_questions):
        """
        Same as your typical PPO data prep: 
        We generate from 'self.ref_model' using the *short* tokenizer/prompt,
        compute old_logprobs, etc.
        """
        def generate_single_fn(prompt, vb):
            return generate_single_response_and_oldlogprob(
                ref_model=self.ref_model,         # old student
                tokenizer=self.tokenizer,         # short-prompt tokenizer
                prompt=prompt,
                verbose=vb
            )
        
        batch_data = prepare_batch_data_for_grpo(
            batch_questions=batch_questions,
            ensure_dict_fn=ensure_dict_mlx,
            generate_single_fn=generate_single_fn,
            calculate_reward=self.calculate_reward,
            G=self.G,
            verbose=self.verbose
        )
        return batch_data

    def train_step(self, batch_data):
        """
        Single-pass training step, merging:
          1) GRPO/PPO update w.r.t. ref_model (old student).
          2) Distillation update w.r.t. teacher_model that sees a *long prompt*.

        We assume we have: 
          data_item["prompt"] => short prompt (for student),
          data_item["long_prompt"] => the full prompt (for teacher),
          OR we compute a separate long prompt on the fly.

        For simplicity, we'll illustrate how to do distillation on the 'response' tokens.
        """
        if not batch_data:
            return 0.0, 0.0
        
        # 1) Flatten data
        all_short_prompts = []
        all_long_prompts = []
        all_responses = []
        all_old_logprobs = []
        all_rewards = []

        for data_item in batch_data:
            # Possibly your data_item includes the short prompt, the long prompt, etc.
            # We'll just illustrate using 'data_item["prompt"]' as short prompt
            # and suppose 'data_item["long_prompt"]' is the full prompt for teacher.
            short_prompt = data_item.get("prompt", "")
            long_prompt = data_item.get("long_prompt", "")  # or build it, e.g. "System ... User: question..."

            responses = data_item["responses"]
            old_logprobs = data_item["old_logprobs"]
            rewards = data_item["rewards"]

            if not responses:
                continue

            for resp, old_lp, rew in zip(responses, old_logprobs, rewards):
                all_short_prompts.append(short_prompt)
                all_long_prompts.append(long_prompt)
                all_responses.append(resp)
                all_old_logprobs.append(old_lp)
                all_rewards.append(rew)

        if not all_responses:
            return 0.0, 0.0

        # 2) Compute advantages (NumPy => MLX)
        advantages_arr = compute_advantages(all_rewards)  # shape [N]
        advantages_m = mx.array(advantages_arr, mx.float32)

        # Convert old logprobs => MLX
        old_logprobs_m = mx.array(all_old_logprobs, mx.float32)

        # 3) We need to create a single big batch of (short_prompt+resp) for the student,
        #    and (long_prompt+resp) for the teacher (distillation).
        #    Then we'll gather log-probs from each. We'll do the typical "gather_logprobs" approach.

        # Convert each short_prompt+response => tokens
        student_input_list = []
        teacher_input_list = []

        # Track max lengths so we can batch them in MLX
        max_len_student = 0
        max_len_teacher = 0

        for shortP, longP, resp in zip(all_short_prompts, all_long_prompts, all_responses):
            student_text = shortP + resp   # e.g. short prompt plus the response
            teacher_text = longP + resp   # teacher sees the big prompt plus same response

            # Encode
            st_tokens = self.tokenizer.encode(student_text)
            if not st_tokens:
                st_tokens = [self.tokenizer.eos_token_id]

            t_tokens = self.tokenizer_long.encode(teacher_text)
            if not t_tokens:
                t_tokens = [self.tokenizer_long.eos_token_id]

            max_len_student = max(max_len_student, len(st_tokens))
            max_len_teacher = max(max_len_teacher, len(t_tokens))

            student_input_list.append(st_tokens)
            teacher_input_list.append(t_tokens)

        # 4) Build the MLX arrays
        N = len(all_responses)
        student_ids_np = np.zeros((N, max_len_student), dtype=np.uint32)
        teacher_ids_np = np.zeros((N, max_len_teacher), dtype=np.uint32)

        for i in range(N):
            s_tokens = student_input_list[i]
            t_tokens = teacher_input_list[i]
            student_ids_np[i, :len(s_tokens)] = s_tokens
            teacher_ids_np[i, :len(t_tokens)] = t_tokens

        student_ids_m = mx.array(student_ids_np, mx.uint32)  # shape [N, max_len_student]
        teacher_ids_m = mx.array(teacher_ids_np, mx.uint32)  # shape [N, max_len_teacher]

        # 5) Define the closure to compute combined loss
        def batch_closure(model_instance):
            # ---- (A) Compute GRPO (PPO) loss vs. ref_model for short prompt => new vs old ratio
            out_current = model_instance(student_ids_m)   # shape [N, seq_len, vocab_size]
            logprobs_current = gather_logprobs(out_current, student_ids_m)  # shape [N]

            out_ref = self.ref_model(student_ids_m)       # shape [N, seq_len, vocab_size]
            kl_values = gather_kl_divergence(out_current, out_ref, student_ids_m)  # shape [N]

            ppo_loss_val = grpo_loss(
                logprobs_current=logprobs_current,
                logprobs_old=old_logprobs_m,
                advantages=advantages_m,
                kl_divergences=kl_values,
                clip_range=0.2,
                kl_coeff=self.kl_coeff,
                reduction="mean"
            )

            # ---- (B) Distillation Loss: teacher (long prompt) vs. new student (short prompt).
            # We'll do a simple KL(teacher||student) or KL(student||teacher). 
            # For example, teacher||student means we interpret teacher as the target distribution.
            out_teacher = self.teacher_model(teacher_ids_m)  # shape [N, T, V]

            # We need to compare distributions *only on the 'response' portion* in principle.
            # But for simplicity, let's do full sequence. 
            # If your sequence alignment is tricky, you might need to isolate the "response tokens."
            
            # We'll do a quick log_softmax for teacher + student:
            teacher_lse = mx.logsumexp(out_teacher, axis=-1, keepdims=True)  # [N, T, 1]
            teacher_logprobs = out_teacher - teacher_lse                     # [N, T, V]

            new_lse = mx.logsumexp(out_current, axis=-1, keepdims=True)      # [N, T, 1]
            new_logprobs = out_current - new_lse                              # [N, T, V]

            # Now teacher_probs = exp(teacher_logprobs)
            teacher_probs = mx.exp(teacher_logprobs)

            # A basic KL(teacher || student):
            #   sum p_teacher(t) * [log p_teacher(t) - log p_student(t)]
            # We'll do it token-by-token, then average. 
            # We'll define a small function below for that. 
            distill_loss_val = kl_divergence(teacher_probs, teacher_logprobs, new_logprobs)

            # Weighted distill
            combined_loss = ppo_loss_val + self.alpha_distill * distill_loss_val
            return combined_loss

        # (C) We'll define a small utility for KL(teacher || student):
        def kl_divergence(p_teacher, logp_teacher, logp_student):
            """
            p_teacher: [N, T, V], probability from teacher
            logp_teacher: [N, T, V], log-prob from teacher
            logp_student: [N, T, V], log-prob from student
            Return scalar (mean over [N, T]).
            """
            # elementwise: p_tch * (logp_tch - logp_std)
            kl_elem = p_teacher * (logp_teacher - logp_student)  # shape [N, T, V]
            # sum over vocab
            kl_sum = mx.sum(kl_elem, axis=-1)  # shape [N, T]
            # average over all tokens in the batch
            kl_mean = mx.mean(kl_sum)
            return kl_mean

        # 6) value_and_grad => compute total loss + grads
        loss_value_and_grad = nn.value_and_grad(self.model, batch_closure)
        batch_loss, grads_dict = loss_value_and_grad(self.model)

        # 7) Apply gradients
        mx.eval(grads_dict)
        self.optimizer.update(self.model, grads_dict)

        # 8) Stats
        mean_loss = float(batch_loss)
        mean_reward = float(np.mean(all_rewards)) if all_rewards else 0.0

        logger.info(color_text(
            f"[DistillGRPO] Single Batch => Loss: {mean_loss:.4f}, Mean Reward: {mean_reward:.4f}",
            YELLOW
        ))
        return mean_loss, mean_reward

    def on_batch_end(self, epoch, batch_idx, loss, reward):
        logger.info(color_text(
            f"[DistillGRPO] E{epoch}B{batch_idx} => Loss: {loss:.4f}, Mean Reward: {reward:.4f}",
            YELLOW
        ))
