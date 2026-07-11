"""ChatML-style chat template. Role markers are plain text; <|endoftext|> is
the end-of-turn / stop token. The same template is used for training and
inference."""

EOT = "<|endoftext|>"

ROLE_HEADERS = {
    "system": "<|system|>\n",
    "user": "<|user|>\n",
    "assistant": "<|assistant|>\n",
}

def _validate(messages):
    if not messages:
        raise ValueError("messages must be a non-empty list")
    for m in messages:
        if "role" not in m or "content" not in m:
            raise ValueError(f"each message needs 'role' and 'content': {m}")
        if m["role"] not in ROLE_HEADERS:
            raise ValueError(f"unknown role '{m['role']}', expected one of {list(ROLE_HEADERS)}")

def _segments(messages, add_generation_prompt):
    for m in messages:
        role, content = m["role"], m["content"]
        header = ROLE_HEADERS[role]
        if role == "assistant":
            yield header, False
            yield f"{content}{EOT}\n", True
        else:
            yield f"{header}{content}\n", False

    if add_generation_prompt:
        yield ROLE_HEADERS["assistant"], False

def format_conversation(messages, add_generation_prompt=False):
    _validate(messages)
    return "".join(text for text, _ in _segments(messages, add_generation_prompt))

def render_for_training(messages):
    if messages[-1]["role"] != "assistant":
        raise ValueError("training examples must end with an assistant turn")
    return format_conversation(messages, add_generation_prompt=False)

def render_for_inference(messages):
    if messages[-1]["role"] == "assistant":
        raise ValueError("inference prompts must end with a user (or system) turn")
    return format_conversation(messages, add_generation_prompt=True)

def encode_conversation(messages, tokenizer, add_generation_prompt=False):
    _validate(messages)
    token_ids, assistant_mask = [], []
    for text, trainable in _segments(messages, add_generation_prompt):
        seg_ids = tokenizer.encode(text, allowed_special={EOT})
        token_ids.extend(seg_ids)
        assistant_mask.extend([trainable] * len(seg_ids))
    return token_ids, assistant_mask

def instruction_entry_to_messages(entry):
    user_content = entry["instruction"]
    if entry.get("input"):
        user_content += f"\n\n{entry['input']}"

    messages = [{"role": "user", "content": user_content}]
    if entry.get("output"):
        messages.append({"role": "assistant", "content": entry["output"]})
    return messages

def format_input(entry):
    instruction_text = (
        "Below is an instruction that describes a task. "
        "Write a response that appropriately completes the request."
        f"\n\n### Instruction:\n{entry['instruction']}"
    )
    input_text = f"\n\n### Input:\n{entry['input']}" if entry["input"] else ""
    return instruction_text + input_text