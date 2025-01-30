# src/inference/stats_collector.py
import torch
from colorama import init, Fore, Style

# Initialise colourama so colours reset automatically after each print
init(autoreset=True)

class StatsCollector:
    """
    A utility class for measuring token counts, timings, and approximate memory usage
    on CPU, CUDA, or MPS devices.
    """
    def __init__(self, device: torch.device):
        self.device = device
        self.prompt_token_count = 0
        self.prompt_time = 0.0
        self.gen_token_count = 0
        self.gen_time = 0.0
        self.peak_mem_bytes = 0

    def reset_peak_memory_stats(self):
        """
        Reset peak memory usage stats if running on CUDA.
        For MPS and CPU, PyTorch does not provide a built-in method to reset peak stats.
        """
        if self.device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()

    def capture_peak_memory(self):
        """
        Capture the peak memory usage if running on CUDA.
        For MPS, PyTorch may not fully track memory usage yet.
        CPU usage remains a best-effort approach.
        """
        if self.device.type == "cuda":
            self.peak_mem_bytes = torch.cuda.max_memory_allocated()

    def record_prompt_stats(self, token_count: int, elapsed_time: float):
        """Record stats for the prompt token encoding phase."""
        self.prompt_token_count = token_count
        self.prompt_time = elapsed_time

    def record_generation_stats(self, token_count: int, elapsed_time: float):
        """Record stats for the generation phase."""
        self.gen_token_count = token_count
        self.gen_time = elapsed_time

    def print_summary(self, generated_text: str):
        """Print a summary of the stats."""
        # Calculate tokens-per-second for prompt
        prompt_tps = (
            self.prompt_token_count / self.prompt_time if self.prompt_time > 0 else 0.0
        )

        # Calculate tokens-per-second for generation
        gen_tps = (
            self.gen_token_count / self.gen_time if self.gen_time > 0 else 0.0
        )

        # Convert peak memory to GB
        peak_mem_gb = self.peak_mem_bytes / (1024 ** 3)

        print(f"{Fore.MAGENTA}{'=' * 10}{Style.RESET_ALL}")
        print(f"{Fore.BLUE}Prompt:{Style.RESET_ALL} {self.prompt_token_count} tokens, "
              f"{prompt_tps:.3f} tokens-per-sec")
        print(f"{Fore.GREEN}Generation:{Style.RESET_ALL} {self.gen_token_count} tokens, "
              f"{gen_tps:.3f} tokens-per-sec")
        print(f"{Fore.RED}Peak memory:{Style.RESET_ALL} {peak_mem_gb:.3f} GB")
        print(f"{Fore.YELLOW}Device:{Style.RESET_ALL} {self.device.type}")
        print(f"{Fore.MAGENTA}{'=' * 10}{Style.RESET_ALL}")
