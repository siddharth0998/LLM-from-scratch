import sys
from pathlib import Path
import pytest

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpt2_chatbot.data import chat_format as cf
from gpt2_chatbot.tokenizer import get_tokenizer, get_eot_id, EOT_ID

@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()

def test_training_render_contains_roles_and_stop():
    messages = [
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
    ]
    text = cf.render_for_training(messages)
    assert "<|user|>" in text
    assert "<|assistant|>" in text
    assert text.rstrip().endswith(cf.EOT)  # assistant turn ends with stop token

def test_inference_render_opens_assistant_turn():
    messages = [{"role": "user", "content": "Hi"}]
    prompt = cf.render_for_inference(messages)
    assert prompt.endswith("<|assistant|>\n")
    assert cf.EOT not in prompt  # model must generate the stop token itself

def test_inference_rejects_trailing_assistant():
    messages = [{"role": "assistant", "content": "oops"}]
    with pytest.raises(ValueError):
        cf.render_for_inference(messages)

def test_training_requires_assistant_last():
    messages = [{"role": "user", "content": "Hi"}]
    with pytest.raises(ValueError):
        cf.render_for_training(messages)

def test_unknown_role_rejected():
    with pytest.raises(ValueError):
        cf.format_conversation([{"role": "robot", "content": "x"}])

def test_encode_mask_aligns_with_assistant_span(tok):
    messages = [
        {"role": "user", "content": "What is 2+2?"},
        {"role": "assistant", "content": "4"},
    ]
    ids, mask = cf.encode_conversation(messages, tok)

    assert len(ids) == len(mask)
    assert any(mask), "at least some assistant tokens must be trainable"
    assert not all(mask), "prompt tokens must be masked out"

    # The trained span should decode to the assistant response + stop token.
    trained_ids = [i for i, m in zip(ids, mask) if m]
    decoded = tok.decode(trained_ids)
    assert decoded == "4" + cf.EOT + "\n"

    # The stop token id must appear in the trained span.
    assert get_eot_id(tok) in trained_ids
    assert EOT_ID == 50256

def test_multi_turn_has_multiple_trained_spans(tok):
    messages = [
        {"role": "system", "content": "You are helpful."},
        {"role": "user", "content": "Hi"},
        {"role": "assistant", "content": "Hello!"},
        {"role": "user", "content": "Bye"},
        {"role": "assistant", "content": "Goodbye!"},
    ]
    ids, mask = cf.encode_conversation(messages, tok)
    assert len(ids) == len(mask)

    # Count contiguous trained runs -> should be 2 (one per assistant turn).
    runs = 0
    prev = False
    for m in mask:
        if m and not prev:
            runs += 1
        prev = m
    assert runs == 2

def test_instruction_entry_conversion():
    entry = {"instruction": "Summarize.", "input": "Long text", "output": "Short."}
    messages = cf.instruction_entry_to_messages(entry)
    assert messages[0]["role"] == "user"
    assert "Long text" in messages[0]["content"]
    assert messages[-1] == {"role": "assistant", "content": "Short."}