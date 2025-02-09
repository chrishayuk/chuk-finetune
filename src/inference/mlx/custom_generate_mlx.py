# src/inference/mlx/custom_generate_mlx.py

# Import samplers from submodules
from inference.mlx.samplers.greedy import greedy_generate
from inference.mlx.samplers.top_k import top_k_generate
from inference.mlx.samplers.top_p import top_p_generate

#################################################
# 2. Multi-sample Generation Helper
#################################################

def top_p_sample_n(
    model,
    tokenizer,
    prompt,
    n=4,
    max_tokens=2000,
    temperature=0.6,
    top_p=0.95
):
    """
    Example function that uses top_p_generate multiple times.
    """
    samples = []
    for _ in range(n):
        gen = top_p_generate(
            model=model,
            tokenizer=tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            top_p=top_p,
            temperature=temperature
        )
        samples.append(gen)
    return samples