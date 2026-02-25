"""Fine-tune from a checkpoint with a low constant learning rate.

Usage:
    python finetune.py <checkpoint> --lr 0.001 --steps 50000 --seed 34 --device cuda
"""
import argparse
import csv
import math
import random
import time
from dataclasses import asdict
from pathlib import Path
from typing import Dict, List, Set, Tuple

import torch
import torch.nn as nn

from src.model import ModelConfig, TinyDecoderLM
from src.data import encode_batch, pair_hash, build_holdout_splits
from src.eval import evaluate_exact_match


def cosine_lr(step, max_steps, base_lr, warmup_steps, min_lr_ratio):
    if step < warmup_steps:
        return base_lr * (step + 1) / max(1, warmup_steps)
    if step >= max_steps:
        return base_lr * min_lr_ratio
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
    min_lr = base_lr * min_lr_ratio
    return min_lr + (base_lr - min_lr) * cosine


def sample_batch(batch_size, gen, reserved_hashes, min_dig=1, max_dig=10):
    a = torch.zeros(batch_size, dtype=torch.int64)
    b = torch.zeros(batch_size, dtype=torch.int64)
    for i in range(batch_size):
        n_dig = int(torch.randint(min_dig, max_dig + 1, (1,), generator=gen).item())
        max_val = 10 ** n_dig
        ai = int(torch.randint(0, max_val, (1,), generator=gen, dtype=torch.int64).item())
        bi = int(torch.randint(0, max_val, (1,), generator=gen, dtype=torch.int64).item())
        while pair_hash(ai, bi) in reserved_hashes:
            ai = int(torch.randint(0, max_val, (1,), generator=gen, dtype=torch.int64).item())
            bi = int(torch.randint(0, max_val, (1,), generator=gen, dtype=torch.int64).item())
        a[i] = ai
        b[i] = bi
    return encode_batch(a, b)


def finetune(ckpt_path: str, lr: float, steps: int, seed: int,
             device: str, eval_interval: int, run_dir: str,
             warmup_steps: int = 500, min_lr_ratio: float = 0.01,
             batch_size: int = 512, val_size: int = 5000):
    run_dir = Path(run_dir)
    run_dir.mkdir(parents=True, exist_ok=True)
    ckpt_dir = run_dir / "checkpoints"
    ckpt_dir.mkdir(exist_ok=True)

    # Load checkpoint
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    cfg = ModelConfig(**ckpt["model_config"])
    model = TinyDecoderLM(cfg).to(device)
    model.load_state_dict(ckpt["model_state"])

    n_params = sum(p.numel() for p in model.parameters())
    print(f"Loaded {ckpt_path}: {n_params} params")
    print(f"Original val_exact: {ckpt.get('val_exact', 'N/A')}")
    print(f"Fine-tuning: lr={lr}, steps={steps}, seed={seed}")

    # Setup data (same holdout as training)
    random.seed(seed)
    torch.manual_seed(seed)

    split_dir = Path("results/data")
    split_dir.mkdir(parents=True, exist_ok=True)
    split_path = split_dir / f"holdout_v{val_size}_t10000_seed{seed}.pt"
    splits = build_holdout_splits(val_size, 10000, seed, split_path)

    reserved_hashes = set()
    for ai, bi in zip(splits["val_a"].tolist(), splits["val_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))
    for ai, bi in zip(splits["test_a"].tolist(), splits["test_b"].tolist()):
        reserved_hashes.add(pair_hash(int(ai), int(bi)))

    val_a, val_b = splits["val_a"], splits["val_b"]
    gen = torch.Generator().manual_seed(seed + 1337)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Metrics
    metrics_path = run_dir / "metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        csv.writer(f).writerow(["step", "train_loss", "val_exact", "val_token_acc", "lr", "elapsed_sec"])

    best_val = 0.0
    t0 = time.time()

    for step in range(steps):
        model.train()
        x, y = sample_batch(batch_size, gen, reserved_hashes)
        x, y = x.to(device), y.to(device)

        lr_now = cosine_lr(step, steps, lr, warmup_steps, min_lr_ratio)
        for pg in optimizer.param_groups:
            pg["lr"] = lr_now

        optimizer.zero_grad(set_to_none=True)
        _, loss = model(x, y)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        if step % eval_interval == 0 or step == steps - 1:
            val_exact, val_tok = evaluate_exact_match(model, val_a, val_b, batch_size, device)
            elapsed = time.time() - t0

            with open(metrics_path, "a", newline="") as f:
                csv.writer(f).writerow([step, loss.item(), val_exact, val_tok, lr_now, elapsed])

            print(f"step={step:>6}  loss={loss.item():.4f}  val_exact={val_exact:.4f}  "
                  f"val_tok={val_tok:.5f}  lr={lr_now:.2e}  t={elapsed:.1f}s")

            if val_exact > best_val:
                best_val = val_exact
                torch.save({
                    "model_config": asdict(cfg),
                    "model_state": model.state_dict(),
                    "step": step,
                    "val_exact": val_exact,
                    "params": n_params,
                }, str(ckpt_dir / "best.pt"))
                print(f"  ** New best: {val_exact:.4f}")

    print(f"\nDone. Best val_exact: {best_val:.4f}")
    return best_val


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint", help="Path to checkpoint to fine-tune from")
    p.add_argument("--lr", type=float, default=0.001)
    p.add_argument("--steps", type=int, default=100000)
    p.add_argument("--seed", type=int, default=34)
    p.add_argument("--device", type=str, default="cuda")
    p.add_argument("--eval-interval", type=int, default=500)
    p.add_argument("--run-dir", type=str, default=None)
    p.add_argument("--warmup-steps", type=int, default=500)
    p.add_argument("--min-lr-ratio", type=float, default=0.01)
    p.add_argument("--batch-size", type=int, default=512)
    args = p.parse_args()

    if args.run_dir is None:
        args.run_dir = f"results/finetune/{Path(args.checkpoint).parent.parent.name}_ft"

    finetune(
        args.checkpoint, args.lr, args.steps, args.seed,
        args.device, args.eval_interval, args.run_dir,
        args.warmup_steps, args.min_lr_ratio, args.batch_size
    )
