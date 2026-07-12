import argparse
import sys
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gpt2_chatbot.inference import ChatBot

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/generation.yaml"))
    ap.add_argument("--checkpoint")
    ap.add_argument("--temperature", type=float)
    ap.add_argument("--top-k", type=int)
    ap.add_argument("--max-new-tokens", type=int)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    if args.checkpoint:
        cfg["checkpoint_path"] = args.checkpoint
    if args.temperature is not None:
        cfg["temperature"] = args.temperature
    if args.top_k is not None:
        cfg["top_k"] = args.top_k
    if args.max_new_tokens is not None:
        cfg["max_new_tokens"] = args.max_new_tokens

    print(f"Loading model from {cfg['checkpoint_path']} ...")
    bot = ChatBot(
        checkpoint_path=cfg["checkpoint_path"],
        base_model=cfg["base_model"],
        context_length=cfg["context_length"],
        temperature=cfg["temperature"],
        top_k=cfg["top_k"],
        max_new_tokens=cfg["max_new_tokens"],
        system_prompt=cfg.get("system_prompt"),
    )
    print(f"Ready on {bot.device}. Type your message ('/reset' to clear, '/exit' to quit).\n")

    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nBye!")
            break
        if not user:
            continue
        if user == "/exit":
            print("Bye!")
            break
        if user == "/reset":
            bot.reset()
            print("(conversation cleared)\n")
            continue

        reply = bot.chat(user)
        print(f"Bot: {reply}\n")


if __name__ == "__main__":
    main()
