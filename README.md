## Chat
If you wish to chat with a model through the interactive chat, you can run the following comman

```bash
uv run inference-cli --chat --model_name "Qwen/Qwen2.5-1.5B-instruct" --max_new_tokens 256 --device cpu
```

### CPU
If you wish to use cpu for inference, you can use the following command

```bash
uv run inference-cli --chat --model_name "Qwen/Qwen2.5-1.5B-instruct" --max_new_tokens 256 --device cpu
```

### MPS
If you are using an apple silicon machine, you can also use MPS

```bash
uv run inference-cli --chat--model_name "Qwen/Qwen2.5-1.5B-instruct" --max_new_tokens 256 --device mps
```

### MLX
MLX is substatially faster than MPS for inference, so it's probably wise to use MLX

```bash
uv run inference-cli --chat --model_name "Qwen/Qwen2.5-1.5B-instruct" --max_new_tokens 256 --device mps
```

## Changing models
You can use any model within huggingface to power chat

### mistral small
The new mistral small models are fully supported

```bash
uv run inference-cli --chat --model_name "Qwen/Qwen2.5-3B-instruct" --max_new_tokens 256 --device mlx
```