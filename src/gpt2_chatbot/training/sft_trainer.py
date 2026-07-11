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

def train_sft(model, train_loader, val_loader, optimizer, device, num_epochs,
              eval_freq=50, eval_iter=5, tokenizer=None, start_context=None,
              warmup_ratio=0.03, min_lr=1e-5, grad_clip=1.0,
              checkpoint_path=None):
    train_losses, val_losses, track_lrs = [], [], []
    tokens_seen, global_step = 0, -1

    base_lr = optimizer.param_groups[0]["lr"]
    total_steps = num_epochs * len(train_loader)
    warmup_steps = int(warmup_ratio * total_steps)
    best_val = float("inf")

    for epoch in range(num_epochs):
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

            tokens_seen += input_batch.numel()

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

                if checkpoint_path is not None and val_loss < best_val:
                    best_val = val_loss
                    save_checkpoint(model, checkpoint_path)
                    print(f"  saved checkpoint (val {val_loss:.3f}) -> {checkpoint_path}")

        if tokenizer is not None and start_context is not None:
            generate_and_print_sample(model, tokenizer, device, start_context)

    return {"train_losses": train_losses, "val_losses": val_losses, "lrs": track_lrs}

def save_checkpoint(model, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), path)

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