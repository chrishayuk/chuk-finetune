# src/inference/mlx/custom_generate_mlx.py

# Import samplers from submodules
from inference.mlx.samplers.greedy import greedy_generate
from inference.mlx.samplers.top_k import top_k_generate
from inference.mlx.samplers.top_p import top_p_generate