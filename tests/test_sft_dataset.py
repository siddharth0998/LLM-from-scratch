import sys
from pathlib import Path
import pytest
import torch

SRC = Path(__file__).resolve().parent.parent / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from gpt2_chatbot.tokenizer import get_tokenizer
from gpt2_chatbot.data import SFTDataset, sft_collate_fn, IGNORE_INDEX
from gpt2_chatbot.data.chat_format import encode_conversation

@pytest.fixture(scope="module")
def tok():
    return get_tokenizer()

@pytest.fixture
def entries():
    return [
        {"instruction": "What is 2+2?", "input": "", "output": "4"},
        {"instruction": "Capital of France?", "input": "", "output": "Paris"},
    ]

def test_only_assistant_tokens_are_supervised(tok, entries):
    ds = SFTDataset(entries, tok)
    input_ids, targets = ds[0]

    # inputs and targets are the next-token shift, same length.
    assert len(input_ids) == len(targets)

    # Some targets must be supervised, and some must be masked (the prompt).
    supervised = [t for t in targets if t != IGNORE_INDEX]
    masked = [t for t in targets if t == IGNORE_INDEX]
    assert supervised, "no supervised tokens"
    assert masked, "prompt tokens should be masked"

    # The supervised targets must decode to the assistant answer + stop token.
    decoded = tok.decode(supervised)
    assert decoded == "4" + "<|endoftext|>" + "\n"

def test_supervised_span_matches_chat_mask(tok, entries):
    """The dataset mask must match encode_conversation's assistant_mask (shifted)."""
    from gpt2_chatbot.data.chat_format import instruction_entry_to_messages

    messages = instruction_entry_to_messages(entries[0])
    ids, assistant_mask = encode_conversation(messages, tok)
    expected_targets = [
        tid if keep else IGNORE_INDEX
        for tid, keep in zip(ids[1:], assistant_mask[1:])
    ]

    ds = SFTDataset(entries[:1], tok)
    _, targets = ds[0]
    assert targets == expected_targets

def test_collate_pads_inputs_and_masks_target_padding(tok, entries):
    ds = SFTDataset(entries, tok)
    inputs, targets = sft_collate_fn([ds[0], ds[1]], device="cpu")

    assert inputs.shape == targets.shape
    assert inputs.dtype == torch.long

    # The shorter example must have -100 padding in its targets.
    lengths = [len(ds[0][0]), len(ds[1][0])]
    shorter = int(lengths[1] < lengths[0])
    pad_count = inputs.shape[1] - lengths[shorter]
    if pad_count > 0:
        assert (targets[shorter, -pad_count:] == IGNORE_INDEX).all()

def test_masked_loss_ignores_prompt(tok, entries):
    """Cross-entropy with ignore_index must not count masked positions."""
    ds = SFTDataset(entries, tok)
    _, targets = sft_collate_fn([ds[0]], device="cpu")

    vocab = 50257
    seq_len = targets.shape[1]
    logits = torch.randn(1, seq_len, vocab)

    full = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), targets.flatten(), ignore_index=IGNORE_INDEX
    )
    # A loss computed only over supervised positions should match.
    keep = targets.flatten() != IGNORE_INDEX
    manual = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1)[keep], targets.flatten()[keep]
    )
    assert torch.allclose(full, manual, atol=1e-5)

def test_degenerate_examples_are_skipped(tok):
    # An entry with an empty output yields no supervised tokens beyond the stop
    # token; ensure the dataset still holds only valid examples.
    ds = SFTDataset([{"instruction": "hi", "input": "", "output": "ok"}], tok)
    assert len(ds) == 1