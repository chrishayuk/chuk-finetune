import argparse
import sys

# imports
from inference.infer import run_inference

def parse_args():
    """Parse command-line arguments for inference."""
    parser = argparse.ArgumentParser(description="Simple CLI for MLX/Torch inference.")
    parser.add_argument(
        "--prompt", type=str, default=None,
        help="User prompt for single-turn mode (if not --chat)."
    )
    parser.add_argument(
        "--chat", action="store_true",
        help="Enable interactive chat mode."
    )
    parser.add_argument(
        "--system_prompt", type=str,
        default="You are a helpful assistant.",
        help="A system prompt giving high-level context."
    )
    parser.add_argument(
        "--model_name", type=str,
        default="Qwen/Qwen2.5-7B-Instruct",
        help="Model name or local path for inference."
    )
    parser.add_argument(
        "--max_new_tokens", type=int,
        default=512,
        help="Maximum number of tokens to generate."
    )
    parser.add_argument(
        "--device", type=str,
        default=None,
        help="Device for inference: cpu, cuda, mps, or mlx."
    )
    return parser.parse_args()

def main():
    # parse arguments
    args = parse_args()

    # get system prompt and device
    system_prompt = args.system_prompt
    device = args.device

    # We maintain a simple conversation history
    user_messages = []
    assistant_messages = []

    if not args.chat:
        # Single-turn mode
        single_prompt = args.prompt or "Who is Ada Lovelace?"
        user_messages.append(single_prompt)

        # Call the unified inference function
        response = run_inference(
            model_name=args.model_name,
            system_prompt=system_prompt,
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            max_new_tokens=args.max_new_tokens,
            device=device
        )
        print(f"\nAssistant: {response}")

    else:
        # Interactive chat mode
        print("Entering chat mode. Type 'exit' or 'quit' to end.\n")
        while True:
            try:
                user_input = input("User: ")
            except (EOFError, KeyboardInterrupt):
                print("\nExiting chat mode.")
                sys.exit(0)

            if user_input.strip().lower() in ["exit", "quit"]:
                print("Exiting chat mode.")
                break

            user_messages.append(user_input)

            # Call the unified inference function
            response = run_inference(
                model_name=args.model_name,
                system_prompt=system_prompt,
                user_messages=user_messages,
                assistant_messages=assistant_messages,
                max_new_tokens=args.max_new_tokens,
                device=device
            )
            cleaned_reply = response.strip()
            if cleaned_reply:
                assistant_messages.append(cleaned_reply)
                print(f"Assistant: {cleaned_reply}\n")
        print("Chat session ended.")

if __name__ == "__main__":
    main()
