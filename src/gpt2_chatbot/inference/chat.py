import torch
from ..model import GPTModel, get_config
from ..tokenizer import (
    get_tokenizer, text_to_token_ids, token_ids_to_text, get_eot_id, EOT_TOKEN,
)
from ..data import render_for_inference
from .generate import generate

MODEL_NAME_MAP = {
    "gpt2": "gpt2-small (124M)",
    "gpt2-medium": "gpt2-medium (355M)",
    "gpt2-large": "gpt2-large (774M)",
    "gpt2-xl": "gpt2-xl (1558M)",
}

class ChatBot:
    def __init__(self, checkpoint_path, base_model="gpt2", context_length=1024,
                 device=None, temperature=1.4, top_k=40, max_new_tokens=256,
                 system_prompt=None):
        self.device = torch.device(device) if device else torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
        self.context_length = context_length
        self.temperature = temperature
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.system_prompt = system_prompt

        self.tokenizer = get_tokenizer()
        self.eos_id = get_eot_id(self.tokenizer)

        preset = MODEL_NAME_MAP[base_model]
        cfg = get_config(preset, context_length=context_length, qkv_bias=True, drop_rate=0.0)
        self.model = GPTModel(cfg)
        self._load_weights(checkpoint_path)
        self.model.to(self.device).eval()

        self.reset()

    def _load_weights(self, path):
        ckpt = torch.load(path, map_location=self.device)
        state = ckpt["model"] if isinstance(ckpt, dict) and "model" in ckpt else ckpt
        self.model.load_state_dict(state)

    def reset(self):
        self.messages = []
        if self.system_prompt:
            self.messages.append({"role": "system", "content": self.system_prompt})

    def chat(self, user_message):
        """Stateful chat: appends the turn to the bot's own running history.

        Use this for a CLI/single-session flow where the ChatBot instance owns
        the conversation. For UIs (e.g. Gradio) that manage their own history
        and support retry/undo, prefer `chat_with_history` to avoid duplicated
        turns accumulating in `self.messages`.
        """
        self.messages.append({"role": "user", "content": user_message})
        reply = self._generate_reply(self.messages)
        self.messages.append({"role": "assistant", "content": reply})
        return reply

    def chat_with_history(self, user_message, history):
        """Stateless chat: rebuild the conversation from `history` each call.

        `history` is the UI-owned conversation (Gradio passes it to the chat
        callback). Rebuilding from it every time keeps retry/undo/edit in sync
        and prevents the same turn from being appended repeatedly, which would
        push the context out of distribution and degrade output quality.
        """
        messages = []
        if self.system_prompt:
            messages.append({"role": "system", "content": self.system_prompt})
        messages.extend(self._history_to_messages(history))
        messages.append({"role": "user", "content": user_message})

        reply = self._generate_reply(messages)

        # Keep internal state in sync with what the UI is showing.
        self.messages = messages + [{"role": "assistant", "content": reply}]
        return reply

    def _generate_reply(self, messages):
        prompt = render_for_inference(messages)
        idx = text_to_token_ids(prompt, self.tokenizer).to(self.device)
        out = generate(
            self.model, idx,
            max_new_tokens=self.max_new_tokens,
            context_size=self.context_length,
            temperature=self.temperature,
            top_k=self.top_k,
            eos_id=self.eos_id,
            use_cache=True,
        )
        full = token_ids_to_text(out, self.tokenizer)
        return full[len(prompt):].replace(EOT_TOKEN, "").strip()

    @staticmethod
    def _history_to_messages(history):
        """Normalize a Gradio chat history into role/content message dicts.

        Handles both supported Gradio formats:
        - "messages": list of {"role", "content"} dicts
        - "tuples":   list of [user_message, assistant_message] pairs
        """
        messages = []
        if not history:
            return messages
        for item in history:
            if isinstance(item, dict):
                role, content = item.get("role"), item.get("content")
                if role in ("user", "assistant") and isinstance(content, str) and content:
                    messages.append({"role": role, "content": content})
            elif isinstance(item, (list, tuple)) and len(item) == 2:
                user_msg, assistant_msg = item
                if isinstance(user_msg, str) and user_msg:
                    messages.append({"role": "user", "content": user_msg})
                if isinstance(assistant_msg, str) and assistant_msg:
                    messages.append({"role": "assistant", "content": assistant_msg})
        return messages