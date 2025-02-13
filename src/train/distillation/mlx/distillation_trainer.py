# src/train/distillation/distillation_trainer.py
import ast
import logging
import numpy as np
import mlx.core as mx
import mlx.nn as nn

from train.trainer_base import Trainer
from train.grpo.grpo_prepare import prepare_batch_data_for_grpo  # We'll reuse the same data prep utility

logger = logging.getLogger(__name__)

def ensure_dict_mlx(item):
    if isinstance(item, dict):
        return item
    if isinstance(item, str) and item.strip().startswith("{"):
        try:
            possible_dict = ast.literal_eval(item)
            if isinstance(item, dict):
                return item
        except:
            pass
    raise ValueError(f"[ERROR] Unexpected non-dict item: {item}")

# Optional for logs
RESET = "\033[0m"
YELLOW = "\033[93m"
def color_text(text, color):
    return f"{color}{text}{RESET}"

class DistillationTrainer(Trainer):
    """
    Phase-1 Distillation:
      - Student sees short prompt
      - Teacher sees long prompt
      - We do KL(teacher || student) on the 'response' tokens
      - We freeze the teacher, update only the student.
    """

    def __init__(
        self,
        student_model,       # trainable student
        teacher_model,       # frozen teacher
        tokenizer_student,   # short prompt tokenizer
        tokenizer_teacher,   # long prompt tokenizer
        optimizer,
        device=None,
        verbose=False
    ):
        super().__init__(student_model, tokenizer_student, optimizer, device=device, verbose=verbose)
        self.teacher_model = teacher_model
        self.tokenizer_teacher = tokenizer_teacher

    def prepare_batch_data(self, batch_questions):
        """
        Optional: We'll reuse 'prepare_batch_data_for_grpo' for convenience,
        even though we're not doing PPO. We just want to produce (prompt, response).
        But if your data already has (prompt, response) pairs, you can skip this.
        
        Suppose each item has:
          - item["prompt"] (short prompt)
          - item["long_prompt"] (long prompt)
          - item["responses"]  (the chain-of-thought or text from teacher)
        We'll assume 'generate_single_response_and_oldlogprob' is not needed here
        unless you want teacher generation on the fly. 
        For a simpler approach, you might already have saved teacher responses.

        We'll do something minimal:
        """
        data_list = []
        for item in batch_questions:
            item = ensure_dict_mlx(item)
            # Suppose item includes:
            #  "prompt": short prompt
            #  "long_prompt": the teacher's bigger system prompt
            #  "responses": list of teacher-generated responses (one or more)
            
            # We'll store them in a format we can handle in train_step
            data_list.append(item)
        return data_list

    def train_step(self, batch_data):
        """
        Single batch => distillation:
          For each item:
            - short_prompt + response => student
            - long_prompt + response => teacher
          Then compute KL(teacher||student).
        """
        if not batch_data:
            return 0.0
        
        all_short_prompts = []
        all_long_prompts = []
        all_responses = []

        for data_item in batch_data:
            short_prompt = data_item.get("prompt", "")
            long_prompt  = data_item.get("long_prompt", "")
            responses    = data_item.get("responses", [])

            # We'll assume there's at least one response 
            # (the teacher's chain-of-thought).
            for resp in responses:
                all_short_prompts.append(short_prompt)
                all_long_prompts.append(long_prompt)
                all_responses.append(resp)

        if not all_responses:
            return 0.0

        N = len(all_responses)

        # Build student inputs
        student_inputs_list = []
        max_len_student = 0

        # Build teacher inputs
        teacher_inputs_list = []
        max_len_teacher = 0

        for shortP, longP, resp in zip(all_short_prompts, all_long_prompts, all_responses):
            # e.g. student sees short prompt + response
            student_text = shortP + resp
            s_tokens = self.tokenizer.encode(student_text)
            if not s_tokens:
                s_tokens = [self.tokenizer.eos_token_id]
            max_len_student = max(max_len_student, len(s_tokens))
            student_inputs_list.append(s_tokens)

            # teacher sees long prompt + response
            teacher_text = longP + resp
            t_tokens = self.tokenizer_teacher.encode(teacher_text)
            if not t_tokens:
                t_tokens = [self.tokenizer_teacher.eos_token_id]
            max_len_teacher = max(max_len_teacher, len(t_tokens))
            teacher_inputs_list.append(t_tokens)

        # Convert to MLX arrays
        student_np = np.zeros((N, max_len_student), dtype=np.uint32)
        teacher_np = np.zeros((N, max_len_teacher), dtype=np.uint32)

        for i in range(N):
            stoks = student_inputs_list[i]
            ttoks = teacher_inputs_list[i]
            student_np[i, :len(stoks)] = stoks
            teacher_np[i, :len(ttoks)] = ttoks

        student_m = mx.array(student_np, mx.uint32)
        teacher_m = mx.array(teacher_np, mx.uint32)

        def distill_closure(model_instance):
            # forward pass student
            out_student = model_instance(student_m)  # shape [N, seq_len, vocab_size]
            # forward pass teacher
            with mx.no_grad():
                out_teacher = self.teacher_model(teacher_m)  # shape [N, T, V]
            
            # log-softmax
            student_lse = mx.logsumexp(out_student, axis=-1, keepdims=True) 
            logp_student = out_student - student_lse
            teacher_lse = mx.logsumexp(out_teacher, axis=-1, keepdims=True)
            logp_teacher = out_teacher - teacher_lse
            p_teacher = mx.exp(logp_teacher)

            # kl(teacher||student) = sum p_teacher * (logp_teacher - logp_student)
            kl_elem = p_teacher * (logp_teacher - logp_student)  # [N, seq_len, V]
            kl_sum = mx.sum(kl_elem, axis=-1)  # [N, seq_len]
            kl_mean = mx.mean(kl_sum)          # scalar
            return kl_mean

        value_and_grad = nn.value_and_grad(self.model, distill_closure)
        batch_loss, grads_dict = value_and_grad(self.model)

        mx.eval(grads_dict)
        self.optimizer.update(self.model, grads_dict)

        float_loss = float(batch_loss)
        logger.info(color_text(f"[DistillationTrainer] Batch => Distill Loss: {float_loss:.4f}", YELLOW))
        return float_loss
