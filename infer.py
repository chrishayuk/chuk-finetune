# infer.py
import argparse
import time
import threading
import torch

from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TextIteratorStreamer
)

# Local utilities
from stats_collector import StatsCollector
from device_selection import DeviceSelector

def parse_args():
    # setup the argument parser
    parser = argparse.ArgumentParser(description="A simple CLI for Qwen inference.")

    # prompt
    parser.add_argument(
        "--prompt",
        type=str,
        default="Give me a short introduction to large language models.",
        help="The user prompt to generate a response for."
    )

    # model name
    parser.add_argument(
        "--model_name",
        type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="The model name or path to use for generation."
    )

    # max tokens
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=512,
        help="The maximum number of new tokens to generate."
    )

    # device (cpu, cuda, mps, or auto)
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Force 'cpu', 'cuda', or 'mps'. If not provided, auto-detects the best available."
    )

    # parse
    return parser.parse_args()

def main():
    # parse arguments
    args = parse_args()

    # Determine which device to use
    device = DeviceSelector.get_preferred_device(args.device)

    # Load the model (no device_map here, since we are forcing it ourselves)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        torch_dtype="auto"
    )
    model.to(device)

    # Create our stats collector
    stats = StatsCollector(device)

    # Load the tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Prepare the messages
    messages = [
        {"role": "user", "content": args.prompt}
    ]

    # Convert to model-friendly text
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True
    )

    # Measure prompt encoding time
    encode_start = time.time()
    # Tokenise and move to the specified device
    model_inputs = tokenizer([text], return_tensors="pt").to(device)
    encode_end = time.time()

    # Number of tokens in the prompt
    prompt_token_count = model_inputs.input_ids.shape[1]
    stats.record_prompt_stats(
        token_count=prompt_token_count,
        elapsed_time=(encode_end - encode_start)
    )

    # Reset memory stats before generation (if applicable)
    stats.reset_peak_memory_stats()

    # Create a streamer to capture tokens as they are generated
    streamer = TextIteratorStreamer(tokenizer, skip_special_tokens=True)

    # We'll measure generation speed by counting tokens as they stream
    gen_token_count = 0

    # Background generation function
    def generate_in_background():
        model.generate(
            **model_inputs,
            max_new_tokens=args.max_new_tokens,
            streamer=streamer
        )

    generation_thread = threading.Thread(target=generate_in_background)

    # Start measuring time right before generation begins
    generation_start = time.time()
    generation_thread.start()

    # We'll collect the streamed text to display afterwards
    generated_text_pieces = []

    # Stream token-by-token
    for new_text in streamer:
        generated_text_pieces.append(new_text)
        # Count how many tokens arrived in this piece
        gen_token_count += len(tokenizer.encode(new_text, add_special_tokens=False))
        print(new_text, end="", flush=True)

    generation_thread.join()
    generation_end = time.time()

    # Capture the peak memory usage
    stats.capture_peak_memory()

    # Combine all streamed pieces
    full_generation = "".join(generated_text_pieces)

    # Record generation stats
    stats.record_generation_stats(
        token_count=gen_token_count,
        elapsed_time=(generation_end - generation_start)
    )

    print()
    # Print summary
    stats.print_summary(full_generation)

if __name__ == "__main__":
    main()
