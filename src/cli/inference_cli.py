import argparse
import sys

from inference.infer import run_inference_flow

def parse_args():
    parser = argparse.ArgumentParser(description="A simple CLI for Qwen inference.")
    parser.add_argument("--prompt", type=str, default=None,
                        help="User prompt for single-turn mode.")
    parser.add_argument("--chat", action="store_true",
                        help="Enable interactive chat mode.")
    parser.add_argument("--system_prompt", type=str,
                        default="You are a helpful assistant.",
                        help="A system prompt giving high-level context.")
    parser.add_argument("--model_name", type=str,
                        default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model name or local path.")
    parser.add_argument("--max_new_tokens", type=int,
                        default=512,
                        help="Maximum number of new tokens to generate.")
    parser.add_argument("--device", type=str,
                        default=None,
                        help="cpu, cuda, or mps. If not provided, auto-detect.")
    return parser.parse_args()

def main():
    args = parse_args()

    if not args.chat:
        # Single-turn usage
        prompt = args.prompt or "Give me a short introduction to large language models."
        response = run_inference_flow(
            model_name=args.model_name,
            system_prompt=args.system_prompt,
            user_messages=[prompt],
            assistant_messages=[],
            max_new_tokens=args.max_new_tokens,
            device_override=args.device
        )
        print(f"\nAssistant: {response}")
    else:
        # Interactive chat mode
        print("Entering chat mode. Type 'exit' or 'quit' (without quotes) to end.\n")

        system_prompt = args.system_prompt
        user_messages = []
        assistant_messages = []

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

            # Stream the new assistant reply, skipping any repeated lines
            new_reply = run_inference_flow(
                model_name=args.model_name,
                system_prompt=system_prompt,
                user_messages=user_messages,
                assistant_messages=assistant_messages,
                max_new_tokens=args.max_new_tokens,
                device_override=args.device
            )

            cleaned_reply = new_reply.strip()
            if cleaned_reply:
                assistant_messages.append(cleaned_reply)
                #print(f"Assistant: {cleaned_reply}\n")
            #else:
                #print("\nNo response was generated.\n")

        print("Chat session ended.")

if __name__ == "__main__":
    main()
