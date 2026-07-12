import os
import sys
from pathlib import Path
import yaml

REPO_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO_ROOT / "src"))

from gpt2_chatbot.inference import ChatBot

def load_bot():
    cfg = yaml.safe_load(open(REPO_ROOT / "configs/generation.yaml"))
    checkpoint = os.environ.get("CHECKPOINT", cfg["checkpoint_path"])
    return ChatBot(
        checkpoint_path=checkpoint,
        base_model=cfg["base_model"],
        context_length=cfg["context_length"],
        temperature=cfg["temperature"],
        top_k=cfg["top_k"],
        max_new_tokens=cfg["max_new_tokens"],
        system_prompt=cfg.get("system_prompt"),
    )

def main():
    import gradio as gr

    bot = load_bot()

    def respond(message, history):
        # Rebuild the conversation from Gradio's own history each call so that
        # retry/undo/edit stay in sync and turns aren't duplicated.
        return bot.chat_with_history(message, history)

    def clear():
        bot.reset()

    with gr.Blocks(title="GPT-2 Chatbot (from scratch)") as demo:
        gr.Markdown("# GPT-2 Chatbot\nFine-tuned from scratch (SFT on Alpaca).")
        chat = gr.ChatInterface(fn=respond)
        chat.chatbot.clear(clear)

    demo.launch()

if __name__ == "__main__":
    main()
