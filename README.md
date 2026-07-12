# GPT-2 Chatbot (from scratch)

A GPT-2 **small (124M)** language model implemented from scratch in PyTorch, then
turned into an instruction-following chatbot via supervised fine-tuning (SFT) on the
[Alpaca](https://github.com/tatsu-lab/stanford_alpaca) dataset. The model architecture
was built and validated in the notebooks, and this repo wraps it in a reusable inference
engine (KV-cache decoding, sampling, a ChatML-style chat template, CLI + web UIs).

> Status: the architecture, training pipeline, and inference engine work end-to-end.
> Because the base model is only 124M parameters and is fine-tuned on a small dataset,
> replies can occasionally be off-topic or degrade into gibberish. See
> [Troubleshooting](#troubleshooting) for the common causes and fixes.

## Features

- **GPT-2 architecture from scratch** — multi-head attention, transformer blocks,
  LayerNorm, and positional embeddings (`src/gpt2_chatbot/model/`).
- **Pretrained weight loading** — loads OpenAI GPT-2 weights via HuggingFace
  `transformers` (`model/weight_loader.py`).
- **KV cache** — single prefill pass then one-token-at-a-time decode
  (`inference/kv_cache.py`, `inference/generate.py`). Cached and uncached paths are
  sampling-equivalent under a fixed seed.
- **Sampling controls** — temperature and top-k, with an EOS stop token.
- **ChatML-style chat template** — shared between training and inference so prompts
  match what the model was fine-tuned on (`data/chat_format.py`).
- **SFT training loop** — warmup + cosine-style LR schedule, gradient clipping,
  periodic eval, resume/best checkpointing (`training/sft_trainer.py`).
- **Two interfaces** — a terminal chat (`scripts/chat.py`) and a Gradio web UI
  (`serving/app_gradio.py`).
- **Dockerized web UI** — a `Dockerfile` and `docker-compose.yml` that serve the Gradio
  chat on port 7860 (CPU-only PyTorch, checkpoint baked in).

## Project structure

```
src/gpt2_chatbot/
  model/        # GPT-2 architecture, config, HF weight loader
  tokenizer/    # tiktoken GPT-2 BPE wrapper + token helpers
  data/         # chat template, SFT/pretrain datasets, collate fns
  inference/    # generate(), KV cache, ChatBot class
  training/     # SFT trainer, loss/eval utils, checkpointing
  serving/      # Gradio web UI
scripts/        # prepare_sft_data, train_sft, evaluate, chat
configs/        # sft.yaml, generation.yaml, model/*
notebooks/      # architecture build + sanity checks + Colab training
tests/          # chat format, SFT dataset, KV-cache equivalence
```

## Installation

Requires Python 3.9+.

```bash
python -m venv .venv
source .venv/bin/activate

# Editable install (pulls core deps from pyproject.toml)
pip install -e .

# Or install from requirements.txt
pip install -r requirements.txt

# Optional extras
pip install -e ".[notebook]"   # numpy, matplotlib, tqdm (notebooks)
pip install -e ".[test]"       # pytest, hypothesis (tests)
```

## Quick start

### 1. Train on Colab (recommended)

Fine-tuning a 124M model needs a GPU, so training runs on Google Colab's free GPU.
Open **`notebooks/03_train_on_colab.ipynb`** in Colab, set the runtime to GPU
(**Runtime → Change runtime type → GPU**), and run the cells top to bottom. The
notebook is self-contained: it clones this repo, installs the dependencies, prepares
the Alpaca data, fine-tunes, and saves the checkpoint to your Google Drive (so a
disconnect doesn't lose progress — just re-run and use the **Resume** cell).

When it finishes, download the trained checkpoint from Drive
(`MyDrive/gpt2-sft/gpt2-sft.pth`) into this repo's `checkpoints/` folder.

### 2. Chat locally

With a trained checkpoint in `checkpoints/gpt2-sft.pth`, install the package
(see [Installation](#installation)) and start chatting.

Terminal chat (`/reset` clears history, `/exit` quits):

```bash
python scripts/chat.py --config configs/generation.yaml
# override sampling on the fly:
python scripts/chat.py --temperature 0.7 --top-k 40 --max-new-tokens 200
```

Web UI (Gradio):

```bash
python -m gpt2_chatbot.serving.app_gradio
# or, after `pip install -e .`
gpt2-chat-web
```

### 3. Evaluate (optional)

```bash
python scripts/evaluate.py --config configs/sft.yaml
```

## Run with Docker

The image serves the Gradio web UI on port **7860** using CPU-only PyTorch. It bakes in
`configs/` and `checkpoints/`, so make sure your trained checkpoint is at
`checkpoints/gpt2-sft.pth` before building.

Using docker compose:

```bash
docker compose up --build
```

Or with plain Docker:

```bash
docker build -t gpt2-chatbot .
docker run --rm -p 7860:7860 gpt2-chatbot
```

Then open http://localhost:7860.

To iterate on configs or swap checkpoints without rebuilding the image, uncomment the
`volumes:` block in `docker-compose.yml` to mount `./configs` and `./checkpoints`.

## Configuration

- `configs/sft.yaml` — training: base model, context length, data dir, batch size,
  epochs, LR schedule, checkpoint paths.
- `configs/generation.yaml` — inference: checkpoint path, `temperature`, `top_k`,
  `max_new_tokens`, and an optional `system_prompt`.

## Testing

```bash
pytest
```

Covers the chat template, the SFT dataset/masking, and cached-vs-uncached generation
equivalence.

## Troubleshooting

**The bot sometimes replies with gibberish or off-topic text.** This is expected to some
degree with a 124M model and a light SFT run. Things that help, roughly in order of
impact:

1. **Lower the sampling temperature.** `configs/generation.yaml` and the `ChatBot`
   default can be fairly high (more random). Try `temperature: 0.7` and `top_k: 40`, or
   `temperature: 0.0` for greedy/deterministic output. High temperature is the most
   common cause of incoherent replies.
2. **Train longer / on more data.** A tiny `--limit` or too few epochs leaves the model
   barely instruction-tuned. Use the full Alpaca set and 2+ epochs.
3. **Keep the prompt template consistent.** Training and inference must use the same
   ChatML template (`data/chat_format.py`). Custom prompts that bypass
   `render_for_inference` will push the model out of distribution.
4. **Watch context length.** Generation stops once the sequence reaches the 1024-token
   context window; very long histories get truncated and quality drops. Use `/reset` in
   the CLI to clear history.
5. **Verify the checkpoint.** Confirm `checkpoint_path` points to your fine-tuned
   weights (`checkpoints/gpt2-sft.pth`), not the raw pretrained base model.

If replies are still poor after tuning sampling, the most reliable fix is a longer SFT
run (or a larger base model such as `gpt2-medium` in `configs/sft.yaml`).

## Notebooks

- `01_architecture.ipynb` — building the GPT-2 architecture from scratch.
- `02_sanity_checks.ipynb` — verifying the model and generation.
- `03_train_on_colab.ipynb` — running SFT training on Colab.

## Acknowledgements

- OpenAI GPT-2 for the pretrained weights.
- Stanford Alpaca for the instruction-tuning dataset.
