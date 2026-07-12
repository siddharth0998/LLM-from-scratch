import torch
import torch.nn as nn
from .block import LayerNorm, TransformerBlock

class GPTModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.tok_emb = nn.Embedding(cfg["vocab_size"], cfg["emb_dim"])
        self.pos_emb = nn.Embedding(cfg["context_length"], cfg["emb_dim"])
        self.drop_emb = nn.Dropout(cfg["drop_rate"])

        self.trf_blocks = nn.Sequential(
            *[TransformerBlock(cfg) for _ in range(cfg["n_layers"])]
        )

        self.final_norm = LayerNorm(cfg["emb_dim"])
        self.out_head = nn.Linear(cfg["emb_dim"], cfg["vocab_size"], bias=False)

    def forward(self, in_idx, cache=None, pos_offset=0):
        # Validate the positional offset before producing any logits.
        context_length = self.pos_emb.weight.shape[0]
        if pos_offset < 0:
            raise ValueError("pos_offset must be non-negative")
        if pos_offset >= context_length:
            raise ValueError(
                f"pos_offset ({pos_offset}) exceeds context length "
                f"({context_length})"
            )

        room = context_length - pos_offset
        batch_size, seq_len = in_idx.shape
        if seq_len > room:
            in_idx = in_idx[:, :room]
            seq_len = room

        tok_embeds = self.tok_emb(in_idx)
        positions = torch.arange(
            pos_offset, pos_offset + seq_len, device=in_idx.device
        )
        pos_embeds = self.pos_emb(positions)
        x = tok_embeds + pos_embeds
        x = self.drop_emb(x)

        if cache is None:
            x = self.trf_blocks(x)
        else:
            for i, block in enumerate(self.trf_blocks):
                past = cache.get(i)
                x, present = block(x, past_kv=past, use_cache=True)
                n_cached = 0 if past is None else past[0].shape[2]
                cache.update(
                    i,
                    present[0][:, :, n_cached:, :],
                    present[1][:, :, n_cached:, :],
                )

        x = self.final_norm(x)
        logits = self.out_head(x)
        return logits