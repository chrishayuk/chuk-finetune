# src/inference/torch/custom_generate_torch.py
# Import samplers from submodules
from inference.torch.samplers.greedy import greedy_generate_torch
from inference.torch.samplers.top_p import top_p_generate_torch

# Optionally define multi-sample or other logic here
def top_p_sample_n_torch(
    model,
    tokenizer,
    prompt,
    n=4,
    max_new_tokens=2000,
    temperature=0.6,
    top_p=0.95
):
    """
    Example function that calls top_p_generate_torch multiple times
    to produce 'n' samples.
    """
    samples = []
    for _ in range(n):
        gen = top_p_generate_torch(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p
        )
        samples.append(gen)
    return samples
