import torch
import torch.nn as nn

class MultiHeadAttention(nn.Module):
    def __init__(self, d_in, d_out, context_length, dropout, num_heads, qkv_bias=False):
        super().__init__()
        assert d_out % num_heads == 0, "d_out must be divisible by num_heads"

        self.d_out = d_out
        self.num_heads = num_heads
        self.head_dim = d_out // num_heads

        self.W_query = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_key = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.W_value = nn.Linear(d_in, d_out, bias=qkv_bias)
        self.out_proj = nn.Linear(d_out, d_out)
        self.dropout = nn.Dropout(dropout)
        self.register_buffer(
            "mask",
            torch.triu(torch.ones(context_length, context_length), diagonal=1),
        )

    def forward(self, x, past_kv=None, use_cache=False):
        b, num_tokens, d_in = x.shape

        # Default path: byte-for-byte identical to the original implementation.
        # Preserved for training and all existing callers.
        if not use_cache and past_kv is None:
            keys = self.W_key(x)
            queries = self.W_query(x)
            values = self.W_value(x)

            keys = keys.view(b, num_tokens, self.num_heads, self.head_dim)
            values = values.view(b, num_tokens, self.num_heads, self.head_dim)
            queries = queries.view(b, num_tokens, self.num_heads, self.head_dim)

            keys = keys.transpose(1, 2)
            queries = queries.transpose(1, 2)
            values = values.transpose(1, 2)

            attn_scores = queries @ keys.transpose(2, 3)
            mask_bool = self.mask.bool()[:num_tokens, :num_tokens]
            attn_scores.masked_fill_(mask_bool, -torch.inf)

            attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
            attn_weights = self.dropout(attn_weights)

            context_vec = (attn_weights @ values).transpose(1, 2)
            context_vec = context_vec.contiguous().view(b, num_tokens, self.d_out)
            context_vec = self.out_proj(context_vec)

            return context_vec

        # Caching path: project Q/K/V for the new tokens only.
        n_new = num_tokens

        keys = self.W_key(x)
        queries = self.W_query(x)
        values = self.W_value(x)

        keys = keys.view(b, n_new, self.num_heads, self.head_dim)
        values = values.view(b, n_new, self.num_heads, self.head_dim)
        queries = queries.view(b, n_new, self.num_heads, self.head_dim)

        keys = keys.transpose(1, 2)
        queries = queries.transpose(1, 2)
        values = values.transpose(1, 2)

        # Concatenate cached K/V (layout [b, num_heads, N_cached, head_dim]).
        if past_kv is not None:
            past_k, past_v = past_kv
            n_cached = past_k.shape[2]
            keys = torch.cat([past_k, keys], dim=2)
            values = torch.cat([past_v, values], dim=2)
        else:
            n_cached = 0

        attn_scores = queries @ keys.transpose(2, 3)
        mask_bool = self.mask.bool()[n_cached:n_cached + n_new, :n_cached + n_new]
        attn_scores.masked_fill_(mask_bool, -torch.inf)

        attn_weights = torch.softmax(attn_scores / keys.shape[-1] ** 0.5, dim=-1)
        attn_weights = self.dropout(attn_weights)

        context_vec = (attn_weights @ values).transpose(1, 2)
        context_vec = context_vec.contiguous().view(b, n_new, self.d_out)
        context_vec = self.out_proj(context_vec)

        return context_vec, (keys, values)
