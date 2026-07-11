import tiktoken
import torch

EOT_TOKEN = "<|endoftext|>"
EOT_ID = 50256

def get_tokenizer(encoding_name="gpt2"):
    return tiktoken.get_encoding(encoding_name)

def get_eot_id(tokenizer=None):
    if tokenizer is not None:
        return tokenizer.encode(EOT_TOKEN, allowed_special={EOT_TOKEN})[0]
    return EOT_ID

def text_to_token_ids(text, tokenizer):
    encoded = tokenizer.encode(text, allowed_special={EOT_TOKEN})
    encoded_tensor = torch.tensor(encoded).unsqueeze(0)
    return encoded_tensor

def token_ids_to_text(token_ids, tokenizer):
    flat = token_ids.squeeze(0)
    return tokenizer.decode(flat.tolist())