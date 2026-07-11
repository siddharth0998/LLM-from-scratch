import argparse
import json
import os
import urllib.request
from pathlib import Path

ALPACA_URL = (
    "https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/"
    "main/alpaca_data.json"
)

REPO_ROOT = Path(__file__).resolve().parent.parent

def download_alpaca(raw_path):
    raw_path = Path(raw_path)
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        print(f"Using cached raw dataset: {raw_path}")
    else:
        print(f"Downloading Alpaca -> {raw_path}")
        urllib.request.urlretrieve(ALPACA_URL, raw_path)
    with open(raw_path, "r", encoding="utf-8") as f:
        return json.load(f)

def split(data, train_frac=0.9, val_frac=0.05):
    n = len(data)
    n_train = int(n * train_frac)
    n_val = int(n * val_frac)
    return {
        "train": data[:n_train],
        "val": data[n_train:n_train + n_val],
        "test": data[n_train + n_val:],
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default=str(REPO_ROOT / "data/raw/alpaca_data.json"))
    ap.add_argument("--out", default=str(REPO_ROOT / "data/processed/alpaca"))
    ap.add_argument("--limit", type=int, default=None, help="Keep only first N entries.")
    args = ap.parse_args()

    data = download_alpaca(args.raw)
    if args.limit:
        data = data[: args.limit]
    print(f"Loaded {len(data)} entries")

    splits = split(data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, rows in splits.items():
        path = out_dir / f"{name}.json"
        with open(path, "w", encoding="utf-8") as f:
            json.dump(rows, f)
        print(f"  {name}: {len(rows):>6} -> {path}")

    print("Done.")

if __name__ == "__main__":
    main()