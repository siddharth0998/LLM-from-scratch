import torch

HF_MODEL_CONFIGS = {
    "gpt2": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

def _assign(param, tensor):
    """Return a Parameter from a torch tensor, validating the shape matches."""
    if tuple(param.shape) != tuple(tensor.shape):
        raise ValueError(f"Shape mismatch. Left: {tuple(param.shape)}, Right: {tuple(tensor.shape)}")
    return torch.nn.Parameter(tensor.detach().clone())

def load_hf_weights_into_gpt(gpt, hf_model=None, model_name="gpt2"):
    """Copy HuggingFace GPT2LMHeadModel weights into gpt (built with qkv_bias=True,
    context_length=1024)."""
    if hf_model is None:
        from transformers import GPT2LMHeadModel
        hf_model = GPT2LMHeadModel.from_pretrained(model_name)

    sd = hf_model.state_dict()
    d = gpt.tok_emb.weight.shape[1]

    gpt.tok_emb.weight = _assign(gpt.tok_emb.weight, sd["transformer.wte.weight"])
    gpt.pos_emb.weight = _assign(gpt.pos_emb.weight, sd["transformer.wpe.weight"])

    for b in range(len(gpt.trf_blocks)):
        p = f"transformer.h.{b}."
        blk = gpt.trf_blocks[b]

        q_w, k_w, v_w = sd[p + "attn.c_attn.weight"].split(d, dim=1)
        q_b, k_b, v_b = sd[p + "attn.c_attn.bias"].split(d, dim=0)
        blk.att.W_query.weight = _assign(blk.att.W_query.weight, q_w.T)
        blk.att.W_key.weight = _assign(blk.att.W_key.weight, k_w.T)
        blk.att.W_value.weight = _assign(blk.att.W_value.weight, v_w.T)
        blk.att.W_query.bias = _assign(blk.att.W_query.bias, q_b)
        blk.att.W_key.bias = _assign(blk.att.W_key.bias, k_b)
        blk.att.W_value.bias = _assign(blk.att.W_value.bias, v_b)

        blk.att.out_proj.weight = _assign(blk.att.out_proj.weight, sd[p + "attn.c_proj.weight"].T)
        blk.att.out_proj.bias = _assign(blk.att.out_proj.bias, sd[p + "attn.c_proj.bias"])

        blk.ff.layers[0].weight = _assign(blk.ff.layers[0].weight, sd[p + "mlp.c_fc.weight"].T)
        blk.ff.layers[0].bias = _assign(blk.ff.layers[0].bias, sd[p + "mlp.c_fc.bias"])
        blk.ff.layers[2].weight = _assign(blk.ff.layers[2].weight, sd[p + "mlp.c_proj.weight"].T)
        blk.ff.layers[2].bias = _assign(blk.ff.layers[2].bias, sd[p + "mlp.c_proj.bias"])

        blk.norm1.scale = _assign(blk.norm1.scale, sd[p + "ln_1.weight"])
        blk.norm1.shift = _assign(blk.norm1.shift, sd[p + "ln_1.bias"])
        blk.norm2.scale = _assign(blk.norm2.scale, sd[p + "ln_2.weight"])
        blk.norm2.shift = _assign(blk.norm2.shift, sd[p + "ln_2.bias"])

    gpt.final_norm.scale = _assign(gpt.final_norm.scale, sd["transformer.ln_f.weight"])
    gpt.final_norm.shift = _assign(gpt.final_norm.shift, sd["transformer.ln_f.bias"])
    gpt.out_head.weight = _assign(gpt.out_head.weight, sd["transformer.wte.weight"])
    return gpt