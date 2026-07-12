import argparse
import json
import sys
from pathlib import Path
import torch
import yaml
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from gpt2_chatbot.model import GPTModel, get_config, load_hf_weights_into_gpt, HF_MODEL_CONFIGS
from gpt2_chatbot.tokenizer import get_tokenizer
from gpt2_chatbot.data import SFTDataset, sft_collate_fn
from gpt2_chatbot.training import train_sft, get_device

MODEL_NAME_MAP = {
    "gpt2": "gpt2-small (124M)",
    "gpt2-medium": "gpt2-medium (355M)",
    "gpt2-large": "gpt2-large (774M)",
    "gpt2-xl": "gpt2-xl (1558M)",
}

def load_split(data_dir, name):
    with open(Path(data_dir) / f"{name}.json", "r", encoding="utf-8") as f:
        return json.load(f)

def build_model(base_model, context_length, device):
    preset = MODEL_NAME_MAP[base_model]
    cfg = get_config(preset, context_length=context_length, qkv_bias=True, drop_rate=0.0)
    model = GPTModel(cfg)
    print(f"Loading pretrained weights from HuggingFace '{base_model}' ...")
    load_hf_weights_into_gpt(model, model_name=base_model)
    return model.to(device), cfg

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(REPO_ROOT / "configs/sft.yaml"))
    ap.add_argument("--smoke", action="store_true", help="Tiny overfit run.")
    ap.add_argument("--resume", action="store_true",
                    help="Resume from the last saved checkpoint if it exists.")
    # Optional overrides (applied in-memory only; the YAML file is NOT modified).
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--max-length", type=int)
    ap.add_argument("--num-epochs", type=int)
    ap.add_argument("--learning-rate", type=float)
    ap.add_argument("--base-model")
    ap.add_argument("--eval-freq", type=int)
    ap.add_argument("--save-every", type=int)
    ap.add_argument("--best-every", type=int)
    args = ap.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    # Apply CLI overrides without writing them back to the config file.
    overrides = {
        "batch_size": args.batch_size,
        "max_length": args.max_length,
        "num_epochs": args.num_epochs,
        "learning_rate": args.learning_rate,
        "base_model": args.base_model,
        "eval_freq": args.eval_freq,
        "save_every": args.save_every,
        "best_every": args.best_every,
    }
    for key, val in overrides.items():
        if val is not None:
            cfg[key] = val
            print(f"override: {key} = {val}")

    torch.manual_seed(cfg["seed"])
    device = get_device()
    print("Device:", device)

    tokenizer = get_tokenizer()
    model, model_cfg = build_model(cfg["base_model"], cfg["context_length"], device)

    train_raw = load_split(cfg["data_dir"], "train")
    val_raw = load_split(cfg["data_dir"], "val")

    if args.smoke:
        train_raw, val_raw = train_raw[:16], val_raw[:8]
        # Smoke defaults, unless the user explicitly overrode them on the CLI.
        if args.num_epochs is None:
            cfg["num_epochs"] = 20
        if args.eval_freq is None:
            cfg["eval_freq"] = 2

    train_ds = SFTDataset(train_raw, tokenizer, max_length=cfg["max_length"])
    val_ds = SFTDataset(val_raw, tokenizer, max_length=cfg["max_length"])
    print(f"Train examples: {len(train_ds)} | Val examples: {len(val_ds)}")

    collate = lambda b: sft_collate_fn(b, allowed_max_length=cfg["max_length"], device=device)
    train_loader = DataLoader(train_ds, batch_size=cfg["batch_size"], shuffle=True,
                              drop_last=True, collate_fn=collate)
    val_loader = DataLoader(val_ds, batch_size=cfg["batch_size"], shuffle=False,
                            drop_last=False, collate_fn=collate)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg["learning_rate"], weight_decay=cfg["weight_decay"]
    )

    resume_from = None
    if args.resume:
        ck = Path(cfg["checkpoint_path"])
        resume_from = str(ck.with_name("last_" + ck.name))

    history = train_sft(
        model, train_loader, val_loader, optimizer, device,
        num_epochs=cfg["num_epochs"], eval_freq=cfg["eval_freq"],
        eval_iter=cfg["eval_iter"], tokenizer=tokenizer,
        start_context="<|user|>\nWhat is the capital of France?\n<|assistant|>\n",
        warmup_ratio=cfg["warmup_ratio"], min_lr=cfg["min_lr"],
        grad_clip=cfg["grad_clip"], checkpoint_path=cfg["checkpoint_path"],
        resume_from=resume_from, save_every=cfg.get("save_every", 0),
        best_every=cfg.get("best_every", 0),
    )

    print("Training complete. Best checkpoint at:", cfg["checkpoint_path"])

    # Save loss history so it can be plotted in the notebook.
    history_path = Path(cfg["checkpoint_path"]).with_name("history.json")
    import json as _json
    _json.dump(history, open(history_path, "w"))
    print(f"Loss history saved to: {history_path}")

    return history

if __name__ == "__main__":
    main()