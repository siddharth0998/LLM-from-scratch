import math
from pathlib import Path
import torch
from .utils import calc_loss_batch, evaluate_model, generate_and_print_sample

def _lr_at_step(step, total_steps, base_lr, warmup_steps, min_lr):
    if warmup_steps > 0 and step < warmup_steps:
        return base_lr * (step + 1) / warmup_steps
    if total_steps <= warmup_steps:
        return base_lr
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    progress = min(1.0, progress)
    return min_lr + 0.5 * (base_lr - min_lr) * (1 + math.cos(math.pi * progress))

def save_checkpoint(path, model, optimizer=None, epoch=0, global_step=-1,
                    best_val=float("inf"), weights_only=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    ckpt = {
        "model": model.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val": best_val,
    }
    if optimizer is not None and not weights_only:
        ckpt["optimizer"] = optimizer.state_dict()
    torch.save(ckpt, path)

def load_checkpoint(path, model, optimizer=None, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    if "model" not in ckpt:  # old format: just weights
        model.load_state_dict(ckpt)
        return {"epoch": 0, "global_step": -1, "best_val": float("inf")}
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    return {
        "epoch": ckpt.get("epoch", 0),
        "global_step": ckpt.get("global_step", -1),
        "best_val": ckpt.get("best_val", float("inf")),
    }

def _last_path(checkpoint_path):
    """Derive the 'latest' checkpoint path next to the 'best' one."""
    p = Path(checkpoint_path)
    return str(p.with_name("last_" + p.name))

def train_sft(model, train_loader, val_loader, optimizer, device, num_epochs,
              eval_freq=50, eval_iter=5, tokenizer=None, start_context=None,
              warmup_ratio=0.03, min_lr=1e-5, grad_clip=1.0,
              checkpoint_path=None, resume_from=None, save_every=0):
    train_losses, val_losses, track_lrs = [], [], []

    base_lr = optimizer.param_groups[0]["lr"]
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)

    start_epoch, global_step, best_val = 0, -1, float("inf")
    if resume_from is not None and Path(resume_from).exists():
        meta = load_checkpoint(resume_from, model, optimizer, device)
        start_epoch = meta["epoch"]
        global_step = meta["global_step"]
        best_val = meta["best_val"]
        print(f"Resumed from {resume_from}: epoch {start_epoch}, step {global_step}, best_val {best_val:.3f}")

    last_path = _last_path(checkpoint_path) if checkpoint_path else None

    for epoch in range(start_epoch, num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            global_step += 1

            lr = _lr_at_step(global_step, total_steps, base_lr, warmup_steps, min_lr)
            for group in optimizer.param_groups:
                group["lr"] = lr

            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            if grad_clip is not None:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            # Mid-epoch evals: monitoring only (no heavy Drive writes here).
            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_lrs.append(lr)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"train {train_loss:.3f} | val {val_loss:.3f} | lr {lr:.2e}"
                )

            # Periodic resumable snapshot (full state) for crash recovery.
            if save_every and last_path is not None and global_step > 0 and global_step % save_every == 0:
                save_checkpoint(last_path, model, optimizer,
                                epoch=epoch, global_step=global_step, best_val=best_val)
                print(f"  saved LAST (step {global_step}) -> {last_path}")

        # --- End of epoch: one eval, then save best (weights-only) if improved.
        _, epoch_val = evaluate_model(model, train_loader, val_loader, device, eval_iter)
        if checkpoint_path is not None and epoch_val < best_val:
            best_val = epoch_val
            save_checkpoint(checkpoint_path, model, epoch=epoch + 1,
                            global_step=global_step, best_val=best_val, weights_only=True)
            print(f"  saved BEST weights (val {epoch_val:.3f}) -> {checkpoint_path}")

        # Resumable snapshot (full state) at epoch boundary.
        if last_path is not None:
            save_checkpoint(last_path, model, optimizer,
                            epoch=epoch + 1, global_step=global_step, best_val=best_val)
            print(f"  saved LAST -> {last_path}")

        if tokenizer is not None and start_context is not None:
            generate_and_print_sample(model, tokenizer, device, start_context)

    return {"train_losses": train_losses, "val_losses": val_losses, "lrs": track_lrs}

def train_model_simple(model, train_loader, val_loader, optimizer, device,
                       num_epochs, eval_freq, eval_iter, start_context, tokenizer):
    train_losses, val_losses, track_tokens_seen = [], [], []
    tokens_seen, global_step = 0, -1

    for epoch in range(num_epochs):
        model.train()
        for input_batch, target_batch in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(input_batch, target_batch, model, device)
            loss.backward()
            optimizer.step()
            tokens_seen += input_batch.numel()
            global_step += 1

            if global_step % eval_freq == 0:
                train_loss, val_loss = evaluate_model(
                    model, train_loader, val_loader, device, eval_iter
                )
                train_losses.append(train_loss)
                val_losses.append(val_loss)
                track_tokens_seen.append(tokens_seen)
                print(
                    f"Ep {epoch + 1} (Step {global_step:06d}): "
                    f"Train loss {train_loss:.3f}, Val loss {val_loss:.3f}"
                )

        generate_and_print_sample(model, tokenizer, device, start_context)

    return train_losses, val_losses, track_tokens_seen