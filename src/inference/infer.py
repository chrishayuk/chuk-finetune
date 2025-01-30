import time
import threading
import re
from transformers import TextIteratorStreamer

from model_utils import load_model_and_tokeniser
from inference.stats_collector import StatsCollector
from inference.chat_template import build_chat_prompt

def strip_role_prefixes(line: str) -> str:
    """
    Strips a leading role label like 'assistant', 'user', or 'system'
    (with optional ':' or newline) from a single line.
    """
    # setup the regex pattern
    pattern = re.compile(r'^(assistant|user|system)\s*:?[\r\n]*', re.IGNORECASE)

    # return the stripped pattern
    return pattern.sub('', line)

def skip_repeated_line(line: str, known_lines: set) -> bool:
    """
    Returns True if 'line' (trimmed) exactly appears in known_lines,
    which represent all previous lines in the conversation 
    (system prompt, user messages, and past assistant messages).
    """
    # strip the known lines
    return line.strip() in known_lines

def filter_and_print_chunk(
    chunk: str,
    known_lines: set,
    first_chunk_printed: list
):
    """
    Splits the chunk by newlines, strips role labels, 
    and then skips any line that exactly matches known conversation lines.
    Prints only the newly generated text.

    If this is the very first chunk that actually contains text to print,
    we also print "Assistant: " as a prefix exactly once.
    """
    # split lines
    lines = chunk.splitlines(keepends=True)
    result = []

    # loop through each line
    for line in lines:
        # strip roles from the line
        no_role = strip_role_prefixes(line)

        # skip repeated lines
        if skip_repeated_line(no_role, known_lines):
            continue

        # add
        result.append(no_role)

    # cleanup
    cleaned = "".join(result)

    # If we haven't yet printed anything this turn, print "Assistant: " once
    if not first_chunk_printed[0] and cleaned.strip():
        print("Assistant: ", end="", flush=True)
        first_chunk_printed[0] = True

    # print
    print(cleaned, end="", flush=True)

def final_cleanup(full_text: str, known_lines: set) -> str:
    """
    After all chunks have been received, remove any leftover role labels and 
    any lines exactly matching known conversation lines from the final text.
    Returns only the truly new assistant text.
    """
    # First remove any leftover role labels at start of lines
    pattern_roles = re.compile(r'^(assistant|user|system)\s*:?[\r\n]+', re.MULTILINE | re.IGNORECASE)
    text_no_roles = pattern_roles.sub('', full_text)

    # Then remove lines that match known_lines exactly
    lines = text_no_roles.splitlines()
    final_lines = []
    for line in lines:
        if line.strip() in known_lines:
            continue
        final_lines.append(line)

    return "\n".join(final_lines).strip()

def prepare_input(
    tokeniser,
    system_prompt: str,
    user_messages: list,
    assistant_messages: list,
    device,
    stats: StatsCollector
):
    """
    Builds and encodes the conversation prompt from system, user, and assistant messages.
    """
    text = build_chat_prompt(
        tokeniser=tokeniser,
        system_prompt=system_prompt,
        user_messages=user_messages,
        assistant_messages=assistant_messages,
        add_generation_prompt=True
    )

    encode_start = time.time()
    model_inputs = tokeniser([text], return_tensors="pt").to(device)
    encode_end = time.time()

    prompt_token_count = model_inputs.input_ids.shape[1]
    stats.record_prompt_stats(
        token_count=prompt_token_count,
        elapsed_time=(encode_end - encode_start)
    )

    stats.reset_peak_memory_stats()
    return model_inputs

def run_inference(
    model,
    model_inputs,
    tokeniser,
    device,
    stats,
    max_new_tokens: int,
    known_lines: set
):
    """
    Streams tokens chunk-by-chunk, filtering out repeated lines in real time,
    while still collecting the raw output for final cleanup.

    We also handle printing "Assistant: " for the first chunk that contains text.
    """
    streamer = TextIteratorStreamer(tokeniser, skip_special_tokens=True)
    gen_token_count = 0
    generated_text_pieces = []

    def _generate_thread():
        model.generate(
            **model_inputs,
            max_new_tokens=max_new_tokens,
            streamer=streamer
        )

    generation_thread = threading.Thread(target=_generate_thread)
    generation_start = time.time()
    generation_thread.start()

    # A mutable flag (single-element list) to track if we've printed "Assistant: " yet
    first_chunk_printed = [False]

    # Stream chunks as they arrive
    for raw_chunk in streamer:
        generated_text_pieces.append(raw_chunk)
        filter_and_print_chunk(raw_chunk, known_lines, first_chunk_printed)
        gen_token_count += len(tokeniser.encode(raw_chunk, add_special_tokens=False))

    generation_thread.join()
    generation_end = time.time()

    # Print a final newline for neatness
    print()

    stats.capture_peak_memory()
    stats.record_generation_stats(
        token_count=gen_token_count,
        elapsed_time=(generation_end - generation_start)
    )

    # Combine raw chunks for final cleaning
    return "".join(generated_text_pieces)

def run_inference_flow(
    model_name: str,
    system_prompt: str,
    user_messages: list,
    assistant_messages: list,
    max_new_tokens: int,
    device_override: str = None
):
    """
    The main routine to:
      1) Load model
      2) Build the prompt
      3) Stream the model's reply (filtered in real time)
      4) Return a final cleaned string for conversation history
    """

    stats = StatsCollector(None)
    model, tokeniser, device = load_model_and_tokeniser(model_name, device_override)
    stats.device = device

    # Prepare the input
    model_inputs = prepare_input(
        tokeniser=tokeniser,
        system_prompt=system_prompt,
        user_messages=user_messages,
        assistant_messages=assistant_messages,
        device=device,
        stats=stats
    )

    # Build a set of all lines from system prompt, user messages, and assistant messages
    # We'll skip any line repeated from these.
    known_lines = set()
    known_lines.add(system_prompt.strip())
    for um in user_messages:
        known_lines.add(um.strip())
    for am in assistant_messages:
        known_lines.add(am.strip())

    # Generate streaming output
    raw_text = run_inference(
        model=model,
        model_inputs=model_inputs,
        tokeniser=tokeniser,
        device=device,
        stats=stats,
        max_new_tokens=max_new_tokens,
        known_lines=known_lines
    )

    # Final cleanup for conversation storage
    final_text = final_cleanup(raw_text, known_lines)

    stats.print_summary(final_text)
    return final_text
