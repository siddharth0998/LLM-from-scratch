"""Cached-vs-uncached generation equivalence tests for the KV cache feature."""

import torch

from gpt2_chatbot.model import GPTModel
from gpt2_chatbot.inference.generate import generate, generate_text_simple
from gpt2_chatbot.inference.kv_cache import KV_Cache


def _tiny_cfg():
    return {
        "vocab_size": 50,
        "context_length": 32,
        "emb_dim": 16,
        "n_heads": 4,
        "n_layers": 3,
        "drop_rate": 0.0,
        "qkv_bias": True,
    }


def _build_model(seed=0):
    torch.manual_seed(seed)
    model = GPTModel(_tiny_cfg())
    model.eval()
    return model


def test_kv_cache_not_nn_module():
    assert not isinstance(KV_Cache(3), torch.nn.Module)


def test_state_dict_keys_unchanged_by_caching():
    model = _build_model()
    before = set(model.state_dict().keys())
    idx = torch.randint(0, 50, (1, 5))
    generate(model, idx, max_new_tokens=6, context_size=32, use_cache=True)
    after = set(model.state_dict().keys())
    assert before == after


def test_cached_logits_match_full_forward():
    # Feeding the whole sequence at once vs prefill+decode must match.
    model = _build_model()
    seq = torch.randint(0, 50, (1, 8))

    full = model(seq)[:, -1, :]

    cache = KV_Cache(len(model.trf_blocks))
    logits = model(seq[:, :5], cache=cache, pos_offset=0)
    for t in range(5, 8):
        logits = model(seq[:, t:t + 1], cache=cache, pos_offset=len(cache))
    cached = logits[:, -1, :]

    assert torch.allclose(full, cached, atol=1e-4, rtol=1e-5)


def test_greedy_generation_identical_with_and_without_cache():
    model = _build_model()
    idx = torch.randint(0, 50, (1, 6))

    out_no = generate(model, idx.clone(), max_new_tokens=12, context_size=32,
                      temperature=0.0, use_cache=False)
    out_yes = generate(model, idx.clone(), max_new_tokens=12, context_size=32,
                       temperature=0.0, use_cache=True)

    assert torch.equal(out_no, out_yes)


def test_simple_generation_identical_with_and_without_cache():
    model = _build_model()
    idx = torch.randint(0, 50, (1, 4))

    out_no = generate_text_simple(model, idx.clone(), max_new_tokens=10,
                                  context_size=32, use_cache=False)
    out_yes = generate_text_simple(model, idx.clone(), max_new_tokens=10,
                                   context_size=32, use_cache=True)

    assert torch.equal(out_no, out_yes)


def test_seeded_sampling_identical_with_and_without_cache():
    model = _build_model()
    idx = torch.randint(0, 50, (1, 5))

    torch.manual_seed(1234)
    out_no = generate(model, idx.clone(), max_new_tokens=15, context_size=32,
                      temperature=0.9, top_k=10, use_cache=False)

    torch.manual_seed(1234)
    out_yes = generate(model, idx.clone(), max_new_tokens=15, context_size=32,
                       temperature=0.9, top_k=10, use_cache=True)

    assert torch.equal(out_no, out_yes)


def test_eos_stops_cached_generation():
    model = _build_model()
    idx = torch.randint(0, 50, (1, 4))
    # Greedy is deterministic; find the first token the model would emit, use it
    # as eos so generation stops immediately (no token appended).
    logits = model(idx)[:, -1, :]
    eos = int(torch.argmax(logits))
    out = generate(model, idx.clone(), max_new_tokens=10, context_size=32,
                   temperature=0.0, eos_id=eos, use_cache=True)
    assert out.shape[1] == idx.shape[1]
