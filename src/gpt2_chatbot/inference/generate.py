import torch

def generate_text_simple(model, idx, max_new_tokens, context_size):

    for _ in range(max_new_tokens):
        # Crop the context to the last ``context_size`` tokens.
        idx_cond = idx[:, -context_size:]

        with torch.no_grad():
            logits = model(idx_cond)  # (batch, n_tokens, vocab_size)

        # Focus on the last time step -> (batch, vocab_size).
        logits = logits[:, -1, :]
        probas = torch.softmax(logits, dim=-1)
        idx_next = torch.argmax(probas, dim=-1, keepdim=True)  # (batch, 1)
        idx = torch.cat((idx, idx_next), dim=1)

    return idx

def generate(model, idx, max_new_tokens, context_size,
             temperature=0.0, top_k=None, eos_id=None):
    for _ in range(max_new_tokens):
        idx_cond = idx[:, -context_size:]
        with torch.no_grad():
            logits = model(idx_cond)
        logits = logits[:, -1, :]
        
        # Optional top-k filtering.
        if top_k is not None:
            top_logits, _ = torch.topk(logits, top_k)
            min_val = top_logits[:, -1]
            logits = torch.where(
                logits < min_val,
                torch.tensor(float("-inf")).to(logits.device),
                logits,
            )

        # Temperature scaling + sampling, or greedy decoding.
        if temperature > 0.0:
            logits = logits / temperature
            probs = torch.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)  # (batch, 1)
        else:
            idx_next = torch.argmax(logits, dim=-1, keepdim=True)  # (batch, 1)

        # Stop early on end-of-sequence.
        if eos_id is not None and idx_next == eos_id:
            break

        idx = torch.cat((idx, idx_next), dim=1)

    return idx