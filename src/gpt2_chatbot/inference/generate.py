import torch
from .kv_cache import KV_Cache

def _sample_next_token(logits, temperature=0.0, top_k=None):
    if top_k is not None:
        top_logits, _ = torch.topk(logits, top_k)
        min_val = top_logits[:, -1]
        logits = torch.where(
            logits < min_val,
            torch.tensor(float("-inf")).to(logits.device),
            logits,
        )

    if temperature > 0.0:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        idx_next = torch.multinomial(probs, num_samples=1)
    else:
        idx_next = torch.argmax(logits, dim=-1, keepdim=True)

    return idx_next

def generate_text_simple(model, idx, max_new_tokens, context_size, use_cache=False):
    if not use_cache:
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]
            idx_next = _sample_next_token(logits, temperature=0.0, top_k=None)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

    # Cached path: prefill once, then decode one token at a time.
    ctx_len = model.pos_emb.weight.shape[0]
    cache = KV_Cache(len(model.trf_blocks))

    idx_cond = idx[:, -context_size:]
    with torch.no_grad():
        logits = model(idx_cond, cache=cache, pos_offset=0)
    logits = logits[:, -1, :]

    for _ in range(max_new_tokens):
        idx_next = _sample_next_token(logits, temperature=0.0, top_k=None)
        idx = torch.cat((idx, idx_next), dim=1)
        if len(cache) >= ctx_len:
            break
        with torch.no_grad():
            logits = model(idx_next, cache=cache, pos_offset=len(cache))
        logits = logits[:, -1, :]

    return idx

def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None, use_cache=False):
    if not use_cache:
        # No-cache path: unchanged full-recompute generation.
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -context_size:]
            with torch.no_grad():
                logits = model(idx_cond)
            logits = logits[:, -1, :]

            idx_next = _sample_next_token(logits, temperature=temperature, top_k=top_k)

            if eos_id is not None and idx_next == eos_id:
                break

            idx = torch.cat((idx, idx_next), dim=1)

        return idx

    # Cached path: a single Prefill pass populates the KV cache, then each
    # Decode_Step feeds only the newest token with the correct position offset.
    # Sampling reuses _sample_next_token so the RNG draw order matches the
    # no-cache path exactly (seeded-sampling equivalence).
    ctx_len = model.pos_emb.weight.shape[0]
    cache = KV_Cache(len(model.trf_blocks))

    idx_cond = idx[:, -context_size:]
    with torch.no_grad():
        logits = model(idx_cond, cache=cache, pos_offset=0)
    logits = logits[:, -1, :]

    for _ in range(max_new_tokens):
        idx_next = _sample_next_token(logits, temperature=temperature, top_k=top_k)

        if eos_id is not None and idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

        # Stop when the accumulated sequence reaches the model context length
        # (the next pos_offset would otherwise reach the context limit).
        if len(cache) >= ctx_len:
            break

        with torch.no_grad():
            logits = model(idx_next, cache=cache, pos_offset=len(cache))
        logits = logits[:, -1, :]

    return idx
    