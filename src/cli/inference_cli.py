import argparse
import sys
from inference.infer import execute_chat_generation

def parse_args():
    """
    Parses command-line arguments and returns them.
    """

    # setup the parser
    parser = argparse.ArgumentParser(description="A simple CLI for inference.")

    # arguments
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
                        help="Device to run inference on: cpu, cuda, mps, or mlx for Apple MLX.")
    
    # parse
    return parser.parse_args()

def execute_inference(model_name: str, system_prompt: str,
                       user_messages: list, assistant_messages: list,
                       max_new_tokens: int, device: str) -> str:
    """
    Executes inference using the appropriate backend based on the device argument.
    """
    if device == "mlx":
        from mlx_lm import load, generate
        model, tokenizer = load(model_name)
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_messages[-1] if user_messages else ""}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return generate(model=model, tokenizer=tokenizer, prompt=prompt, max_tokens=max_new_tokens, verbose=True)
    else:
        return execute_chat_generation(
            model_name=model_name,
            system_prompt=system_prompt,
            user_messages=user_messages,
            assistant_messages=assistant_messages,
            max_new_tokens=max_new_tokens,
            device_override=device
        )

def main():
    """
    Main entry point for the CLI.
    """
    args = parse_args()
    system_prompt = args.system_prompt
    user_messages = []
    assistant_messages = []
    
    if not args.chat:
        prompt = args.prompt or "Who is Ada Lovelace?"
        response = execute_inference(
            model_name=args.model_name,
            system_prompt=system_prompt,
            user_messages=[prompt],
            assistant_messages=[],
            max_new_tokens=args.max_new_tokens,
            device=args.device
        )
        print(f"\nAssistant: {response}")
    else:
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
            response = execute_inference(
                model_name=args.model_name,
                system_prompt=system_prompt,
                user_messages=user_messages,
                assistant_messages=assistant_messages,
                max_new_tokens=args.max_new_tokens,
                device=args.device
            )
            cleaned_reply = response.strip()
            if cleaned_reply:
                assistant_messages.append(cleaned_reply)
                print(f"Assistant: {cleaned_reply}\n")
        print("Chat session ended.")

if __name__ == "__main__":
    main()
