import argparse
import json
import sys
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gpt2_chatbot.model import GPTModel, get_config
from gpt2_chatbot.tokenizer import (
    get_tokenizer, text_to_token_ids, token_ids_to_text, get_eot_id, EOT_TOKEN,
)
from gpt2_chatbot.data import SFTDataset, sft_collate_fn, render_for_inference, instruction_entry_to_messages
from gpt2_chatbot.inference import generate
from gpt2_chatbot.training import load_checkpoint, calc_loss_loader, get_device

MODEL_NAME_MAP = {
    "gpt2": "gpt2-small (124M)",
    "gpt2-medium": "gpt2-medium (355M)",
    "gpt2-large": "gpt2-large (774M)",
    "gpt2-xl": "gpt2-xl (1558M)",
}

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/sft.yaml"))
    ap.add_argument("--checkpoint", help="Override the checkpoint path from the config.")
    ap.add_argument("--num-samples", type=int, default=5, help="Sample generations to print.")
    args = ap.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    ckpt_path = args.checkpoint or cfg["checkpoint_path"]

    device = get_device()
    print("Device:", device)
    tokenizer = get_tokenizer()

    # Build model matching the base and load the fine-tuned weights.
    preset = MODEL_NAME_MAP[cfg["base_model"]]
    model_cfg = get_config(preset, context_length=cfg["context_length"], qkv_bias=True, drop_rate=0.0)
    model = GPTModel(model_cfg)
    load_checkpoint(ckpt_path, model, device=device)
    model.to(device).eval()
    print(f"Loaded checkpoint: {ckpt_path}")

    # Test loss (same masked cross-entropy as training).
    test_raw = json.load(open(Path(cfg["data_dir"]) / "test.json"))
    test_ds = SFTDataset(test_raw, tokenizer, max_length=cfg["max_length"])
    collate = lambda b: sft_collate_fn(b, allowed_max_length=cfg["max_length"], device=device)
    test_loader = DataLoader(test_ds, batch_size=cfg["batch_size"], shuffle=False, collate_fn=collate)

    test_loss = calc_loss_loader(test_loader, model, device)
    print(f"\nTest examples: {len(test_ds)}")
    print(f"Test loss (masked): {test_loss:.4f}")

    # Sample generations on unseen test prompts.
    print(f"\n--- {args.num_samples} sample generations ---")
    eos_id = get_eot_id(tokenizer)
    for entry in test_raw[: args.num_samples]:
        messages = instruction_entry_to_messages({**entry, "output": ""})[:1]
        prompt = render_for_inference(messages)
        ids = text_to_token_ids(prompt, tokenizer).to(device)
        out = generate(model, ids, max_new_tokens=200, context_size=model_cfg["context_length"],
                       top_k=40, temperature=0.7, eos_id=eos_id)
        response = token_ids_to_text(out, tokenizer)[len(prompt):]
        response = response.replace(EOT_TOKEN, "").strip()
        print(f"\nInstruction: {entry['instruction']}")
        if entry.get("input"):
            print(f"Input: {entry['input']}")
        print(f"Model: {response}")
        print(f"Reference: {entry['output']}")
        print("-" * 60)

if __name__ == "__main__":
    main()