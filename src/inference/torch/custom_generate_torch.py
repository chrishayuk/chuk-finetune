# src/inference/torch/custom_generate_torch.py
# Import samplers from submodules
from inference.torch.samplers.greedy import greedy_generate_torch
from inference.torch.samplers.top_p import top_p_generate_torch
from inference.torch.samplers.top_p_with_kvcache import top_p_generate_torch_with_kvcache
