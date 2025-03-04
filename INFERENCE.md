# Inference
This CLI allows you to interact with a model allowing you to test inference in a similar fashion on how you would do it using Reinforcement Learning Training.   By default in the samples, i'll use the same settings that DeepSeek used to generate their long chains of thoughts (with some caveats)

## Base Model
By default, i'm using the "Qwen/Qwen2.5-3B" base model for training.

## Greedy
If you wish to use greedy prompting, i.e. top probablistic response

```bash
uv run inference-cli --model_name "Qwen/Qwen2.5-3B" --max_new_tokens 256 --device mlx --prompt "hi"
```

## device frameworks
devices can 

- mlx (apple silicon)
- mps (apple silicon on pytorch)
- cuda (cuda on pytorch)
- cpu (cpu on pytorch)

## Deep Seek Prompt and Settings
This is the slightly older version using the same prompt that deepseek used.
Please note that top-p sampling (of 95% percentile) with temperature of 0.7
I've limited the tokens to 256

```bash
uv run inference-cli --sampler top_p --temperature 0.6 --top_p 0.95 --model_name "Qwen/Qwen2.5-3B" --max_new_tokens 256 --device mlx --prompt "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: What's 10 + 10? Assistant: <think>"
```

This version is slightly simpler, but does just a good of a job with qwen but not as consistent

```bash
uv run inference-cli --sampler top_p --temperature 0.6 --top_p 0.95 --model_name "Qwen/Qwen2.5-3B" --max_new_tokens 256 --device mlx --system_prompt "You are a helpful assistant. Always use <think> for reasoning </think> and <answer> for final answer.</answer>" --prompt "What's 10 + 10? Assistant: <think>"
```

## Generating Multiple Responses
If you wish to generate multiple responses then you can use the following

```bash
uv run inference-cli --sampler top_p --temperature 0.6 --top_p 0.95 --model_name "Qwen/Qwen2.5-3B" --max_new_tokens 256 --device mlx --prompt "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., <think> reasoning process here </think>
<answer> answer here </answer>. User: What's 10 + 10? Assistant: <think>" --num_responses 4
```

if you wish to test the verifier answer prompt:

```bash
uv run inference-cli --sampler top_p --temperature 0.6 --top_p 0.95 --model_name "Qwen/Qwen2.5-3B" --max_new_tokens 256 --device mlx --prompt "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. The assistant first thinks about the reasoning process in the think tags and then provides the user with the answer in a user friendly manner within the answer tags, and finally also provides just the answer within the verifier tags, so it can be checked by an automated process. The reasoning process, answer and verifier answer are enclosed within <think> </think> <answer> </answer> and <verifier_answer> </verifier_answer> tags, respectively, i.e., <think>reasoning process here</think><answer>user answer here</answer><verifier_answer>verifier answer here</verifier_answer>. User: What's 10 + 10? Assistant: <think>"
```

