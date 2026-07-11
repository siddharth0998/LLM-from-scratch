from copy import deepcopy

GPT_CONFIG_124M = {
    "vocab_size": 50257,
    "context_length": 1024,
    "emb_dim": 768,
    "n_heads": 12,
    "n_layers": 12,
    "drop_rate": 0.1,
    "qkv_bias": False,
}

MODEL_CONFIGS = {
    "gpt2-small (124M)": {"emb_dim": 768, "n_layers": 12, "n_heads": 12},
    "gpt2-medium (355M)": {"emb_dim": 1024, "n_layers": 24, "n_heads": 16},
    "gpt2-large (774M)": {"emb_dim": 1280, "n_layers": 36, "n_heads": 20},
    "gpt2-xl (1558M)": {"emb_dim": 1600, "n_layers": 48, "n_heads": 25},
}

def get_config(model_name="gpt2-small (124M)", **overrides):
    if model_name not in MODEL_CONFIGS:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: {list(MODEL_CONFIGS)}")
    cfg = deepcopy(GPT_CONFIG_124M)
    cfg.update(MODEL_CONFIGS[model_name])
    cfg.update(overrides)
    return cfg
