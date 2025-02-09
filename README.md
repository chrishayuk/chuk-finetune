# Inference
The following gives a brief explanation on how to use inference 

## Chat
If you wish to chat with a model through the interactive chat, you can run the following comman

```bash
uv run inference-cli --chat --model_name "Qwen/Qwen2.5-1.5B-instruct" --max_new_tokens 256 --device cpu
```

### Sampling
If you wish to test out prompts for sampling, see the inference.md file for more details
However the following command shows you how to generate multiple samples

```bash
uv run inference-cli --sampler top_p --temperature 0.6 --top_p 0.95 --model_name "Qwen/Qwen2.5-3B" --max_new_tokens 256 --device mlx --prompt "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: What's 10 + 10? Assistant: <think>" --num_responses 4
```

if you wish to test the verifier answer prompt:

```bash
uv run inference-cli --sampler top_p --temperature 0.6 --top_p 0.95 --model_name "Qwen/Qwen2.5-3B" --max_new_tokens 256 --device mlx --prompt "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the think tags and then provides the user with the answer in a user friendly manner within the answer tags, and finally also provides just the answer within the verifier tags, so it can be checked by an automated process. The reasoning process, answer and verifier answer are enclosed within <think> </think> <answer> </answer> and <verifier_answer> </verifier_answer> tags, respectively, i.e., <think>reasoning process here</think><answer>user answer here</answer><verifier_answer>verifier answer here</verifier_answer>. User: What's 10 + 10? Assistant: <think>"
```

## Training

```bash
uv run main.py --device mlx --model Qwen/Qwen2.5-3B
```

or

```bash
uv run main.py --device mps --model Qwen/Qwen2.5-3B
```