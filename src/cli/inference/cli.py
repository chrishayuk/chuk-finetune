#!/usr/bin/env python3
# src/cli/inference/cli.py
import sys
import logging

from cli.inference.arg_parser import parse_arguments
from model.model_loader import load_model_and_tokenizer
from inference.infer import run_inference

logging.basicConfig(
    level=logging.INFO,
    format="%(message)s",
    datefmt="%H:%M:%S",
    stream=sys.stdout
)
logger = logging.getLogger(__name__)

############################################################
# ANSI color codes
############################################################
ANSI_RESET = "\033[0m"
ANSI_BLUE = "\033[94m"
ANSI_GREEN = "\033[92m"
ANSI_YELLOW = "\033[93m"

def extract_last_assistant_block(raw_text: str) -> str:
    """
    Finds the last line "assistant" (case-insensitive) 
    and returns everything after that line exactly as is.
    If not found, returns raw_text unchanged.
    """
    lines = raw_text.splitlines()
    assistant_start_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "assistant":
            assistant_start_idx = i
    if assistant_start_idx is not None:
        # Return everything after the 'assistant' line, preserving all spacing/newlines
        return "\n".join(lines[assistant_start_idx+1:])
    # If we never saw a line with just 'assistant'
    return raw_text

def color_print(role_color, role_label, text=""):
    print(f"{role_color}{role_label}{ANSI_RESET} {text}")

def main():
    # parse arguments
    args = parse_arguments()

    # Convert the comma-separated string to a list, or empty if none
    if args.stop_sequences.strip():
        stop_seqs = [s.strip() for s in args.stop_sequences.split(",")]
    else:
        stop_seqs = []

    # set the default system prompt
    if args.chat and args.system_prompt is None:
        args.system_prompt = "You are a helpful assistant."

    # set the system prompt
    system_prompt = args.system_prompt

    # set the device
    device = args.device

    # load the model and the tokenizer
    logger.info("Loading model: %s (device=%s)", args.model_name, device or "auto")
    model, tokenizer, is_mlx = load_model_and_tokenizer(args.model_name, device)

    # clear messages
    user_messages = []
    assistant_messages = []

    # use the chat template, if chat specified
    use_chat_template = args.chat

    # we don't support streaming for now
    stream_mode = False # getattr(args, "stream", False)

    # check if chat mode
    if not args.chat:
        # Single turn
        single_prompt = args.prompt or "Who is Ada Lovelace?"
        user_messages.append(single_prompt)

        color_print(ANSI_BLUE, "System:", system_prompt)
        print()
        color_print(ANSI_GREEN, "User:", single_prompt)

        # run inference
        response_or_generator = run_inference(
            model=model,
            tokenizer=tokenizer,
            is_mlx=is_mlx,
            system_prompt=system_prompt,
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            max_new_tokens=args.max_new_tokens,
            sampler=args.sampler,
            temperature=args.temperature,
            top_p=args.top_p,
            stop_sequences=stop_seqs,
            use_chat_template=use_chat_template,
            stream=stream_mode,
            num_responses=args.num_responses
        )

        print()  # blank line

        # If run_inference returned multiple responses (a list):
        if isinstance(response_or_generator, list):
            # Print each response with "Assistant (Sample X/Y)"
            for i, resp_text in enumerate(response_or_generator, start=1):
                final_text = extract_last_assistant_block(resp_text)
                color_print(ANSI_YELLOW,
                    f"Assistant (Sample {i}/{len(response_or_generator)}):",
                    final_text
                )
                print()  # extra blank line
        else:
            # Single response path
            if hasattr(response_or_generator, "__iter__") and not isinstance(response_or_generator, str):
                # streaming generator
                partial_chunks = []
                for chunk in response_or_generator:
                    print(chunk, end="", flush=True)
                    partial_chunks.append(chunk)
                print()
                raw_text = "".join(partial_chunks)
                final_text = extract_last_assistant_block(raw_text)
                color_print(ANSI_YELLOW, "Assistant:", final_text)
            else:
                # final single string
                final_text = extract_last_assistant_block(response_or_generator)
                color_print(ANSI_YELLOW, "Assistant:", final_text)

    else:
        # Chat mode
        print("Entering chat mode. Type 'exit' or 'quit' to end.\n")

        # print the system prompt
        color_print(ANSI_BLUE, "System:", system_prompt)
        print()

        # chat loop
        while True:
            try:
                # get the user prompt
                user_input = input(f"{ANSI_GREEN}User:{ANSI_RESET} ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat mode.")
                sys.exit(0)

            # check if the user wants to quit
            if user_input.strip().lower() in ["exit", "quit"]:
                # quit
                print("Exiting chat mode.")
                break

            # add the user prompt to the context
            user_messages.append(user_input)

            # run inferences
            response_or_generator = run_inference(
                model=model,
                tokenizer=tokenizer,
                is_mlx=is_mlx,
                system_prompt=system_prompt,
                user_messages=user_messages,
                assistant_messages=assistant_messages,
                max_new_tokens=args.max_new_tokens,
                sampler=args.sampler,
                temperature=args.temperature,
                top_p=args.top_p,
                stop_sequences=stop_seqs,
                use_chat_template=use_chat_template,
                stream=stream_mode
            )

            # check if we're streaming
            if hasattr(response_or_generator, "__iter__") and not isinstance(response_or_generator, str):
                # It's a generator => streaming
                partial_chunks = []
                for chunk in response_or_generator:
                    print(chunk, end="", flush=True)
                    partial_chunks.append(chunk)
                print()

                raw_text = "".join(partial_chunks)
                final_text = extract_last_assistant_block(raw_text)
            else:
                # final string
                final_text = extract_last_assistant_block(response_or_generator)

            # add the final text to the assistant message
            assistant_messages.append(final_text)

            # print the assistant message
            color_print(ANSI_YELLOW, "Assistant:", final_text)
            print()

        # chat ended
        print("Chat session ended.")


if __name__ == "__main__":
    main()
