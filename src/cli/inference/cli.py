#!/usr/bin/env python3
# src/cli/inference/cli.py
import sys
import logging

# cli imports
from cli.inference.arg_parser import parse_arguments

# inference imports
from inference.infer import load_inference_model, run_inference

# set up the logger
logger = logging.getLogger(__name__)

def main():
    # parse arguments
    args = parse_arguments()

    # set the system prompt and device
    system_prompt = args.system_prompt
    device = args.device

    # 1) Load the model once at the start (Torch or MLX)
    logger.info("Loading model: %s (device=%s)", args.model_name, device or "auto")
    model, tokenizer, is_mlx = load_inference_model(args.model_name, device)

    # We'll keep track of the conversation history
    user_messages = []
    assistant_messages = []

    # are we in chat mode
    if not args.chat:
        # Single-turn mode
        single_prompt = args.prompt or "Who is Ada Lovelace?"
        user_messages.append(single_prompt)

        # 2) Use run_inference with the preloaded model
        response = run_inference(
            model=model,
            tokenizer=tokenizer,
            is_mlx=is_mlx,
            system_prompt=system_prompt,
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            max_new_tokens=args.max_new_tokens
        )

        # print the assistant response
        print(f"\nAssistant: {response}")
    else:
        # Chat mode
        print("Entering chat mode. Type 'exit' or 'quit' to end.\n")

        # chat loop
        while True:
            try:
                # get the user input
                user_input = input("User: ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat mode.")
                sys.exit(0)

            # check if the user wants to quite
            if user_input.strip().lower() in ["exit", "quit"]:
                # quitting
                print("Exiting chat mode.")
                break

            # add the user message to the chat history
            user_messages.append(user_input)
            
            # run inference
            response = run_inference(
                model=model,
                tokenizer=tokenizer,
                is_mlx=is_mlx,
                system_prompt=system_prompt,
                user_messages=user_messages,
                assistant_messages=assistant_messages,
                max_new_tokens=args.max_new_tokens
            )

            # clean up the replay
            cleaned_reply = response.strip()

            # check we have a response
            if cleaned_reply:
                # add the response to the assistant messages
                assistant_messages.append(cleaned_reply)
                print(f"Assistant: {cleaned_reply}\n")

        # done
        print("Chat session ended.")

if __name__ == "__main__":
    main()
