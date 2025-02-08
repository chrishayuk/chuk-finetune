#!/usr/bin/env python3
# src/cli/inference/cli.py
import sys
import logging

# imports
from cli.inference.arg_parser import parse_arguments
from model.model_loader import load_model_and_tokenizer
from inference.infer import run_inference

# logging
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
    and returns everything after that line. 
    If not found, returns raw_text.
    """
    lines = raw_text.splitlines()
    assistant_start_idx = None
    for i, line in enumerate(lines):
        if line.strip().lower() == "assistant":
            assistant_start_idx = i
    if assistant_start_idx is not None:
        return "\n".join(lines[assistant_start_idx+1:]).strip()
    return raw_text.strip()

def main():
    # parse arguments
    args = parse_arguments()

    # get the system prompt and device
    system_prompt = args.system_prompt
    device = args.device

    # load the model
    logger.info("Loading model: %s (device=%s)", args.model_name, device or "auto")
    model, tokenizer, is_mlx = load_model_and_tokenizer(args.model_name, device)

    # clear the user and assistant messages
    user_messages = []
    assistant_messages = []

    # If we want to forcibly stop generation on 'User:' or 'Assistant:', define stop seq:
    stop_seqs = ["User:", "Assistant:"] if args.chat else None

    # If we're in chat mode, we'll attempt to use the chat template
    use_chat_template = args.chat

    # We define a small helper to color-code our prints
    def color_print(role_color, role_label, text=""):
        # e.g. color_print(ANSI_BLUE, "System:", "You are a helpful assistant.")
        print(f"{role_color}{role_label}{ANSI_RESET} {text}")

    if not args.chat:
        # Single turn
        single_prompt = args.prompt or "Who is Ada Lovelace?"
        user_messages.append(single_prompt)

        # Show system prompt once (in blue)
        color_print(ANSI_BLUE, "System:", system_prompt)
        print()  # blank line

        # Show user (in green)
        color_print(ANSI_GREEN, "User:", single_prompt)

        # Inference
        response = run_inference(
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
            # use the chat template only if chat is True
            use_chat_template=use_chat_template
        )

        # extract final assistant block
        final_text = extract_last_assistant_block(response)

        print()
        color_print(ANSI_YELLOW, "Assistant:", final_text)

    else:
        # Chat mode
        print("Entering chat mode. Type 'exit' or 'quit' to end.\n")

        # Show system prompt once (blue)
        color_print(ANSI_BLUE, "System:", system_prompt)
        print()

        while True:
            try:
                # user input in green
                user_input = input(f"{ANSI_GREEN}User:{ANSI_RESET} ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat mode.")
                sys.exit(0)

            # check for user quitting
            if user_input.strip().lower() in ["exit", "quit"]:
                # quitting
                print("Exiting chat mode.")
                break

            user_messages.append(user_input)

            # Inference
            response = run_inference(
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
                use_chat_template=use_chat_template
            )

            # extract final assistant block
            final_text = extract_last_assistant_block(response)

            # store 
            assistant_messages.append(final_text)
            color_print(ANSI_YELLOW, "Assistant:", final_text)
            print()

        print("Chat session ended.")

if __name__ == "__main__":
    main()
