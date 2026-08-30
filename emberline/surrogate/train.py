"""Surrogate training: BCE + soft-Dice, world-split val, checkpoints, resume.

Loss: per-pixel BCE on both output planes plus soft Dice on the ``touched``
plane. BCE alone under-weights the thin newly-burned ring (class imbalance:
most crop pixels are trivially unchanged); Dice directly optimises the
overlap metric we report (IoU is monotone in Dice), so the pair trains
sharp, well-calibrated fronts.

Checkpointing: ``latest.pt`` every ``checkpoint_every`` steps (model, optim,
step, best metric, torch RNG state) and ``best.pt`` whenever val IoU
improves. ``--resume`` (default on) picks up ``latest.pt`` if present, so an
interrupted session continues cleanly. Early stopping after
``early_stop_patience`` val checks without improvement.
"""

from __future__ import annotations

import argparse
import pathlib
import time
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from ..config import load_config, repo_root
from .datasets import FirePairDataset, load_records, split_world_ids
from .model import FireUNet


def dice_loss(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    p = torch.sigmoid(logits)
    inter = (p * target).sum(dim=(1, 2))
    denom = p.sum(dim=(1, 2)) + target.sum(dim=(1, 2))
    return (1 - (2 * inter + 1.0) / (denom + 1.0)).mean()


def batch_iou(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = (torch.sigmoid(logits) > 0.5).float()
    inter = (pred * target).sum().item()
    union = ((pred + target) > 0).float().sum().item()
    return inter / union if union > 0 else 1.0


def loss_fn(logits: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    bce = F.binary_cross_entropy_with_logits(logits, y)
    return bce + dice_loss(logits[:, 0], y[:, 0])


def ckpt_dir(cfg: dict[str, Any]) -> pathlib.Path:
    d = repo_root() / cfg["surrogate"]["train"]["dir"]
    d.mkdir(parents=True, exist_ok=True)
    return d


def save_ckpt(path: pathlib.Path, model, opt, step: int, best_iou: float) -> None:
    torch.save({
        "model": model.state_dict(),
        "opt": opt.state_dict(),
        "step": step,
        "best_iou": best_iou,
        "torch_rng": torch.get_rng_state(),
    }, path)


@torch.no_grad()
def validate(model: FireUNet, loader: DataLoader) -> float:
    model.eval()
    ious = []
    for x, y in loader:
        logits = model(x)
        ious.append(batch_iou(logits[:, 0], y[:, 0]))
    model.train()
    return float(np.mean(ious)) if ious else 0.0


def train(cfg: dict[str, Any], resume: bool = True, max_steps: int | None = None,
          max_minutes: float | None = None) -> pathlib.Path:
    """Train the surrogate; returns path to best checkpoint."""
    torch.set_num_threads(max(4, (torch.get_num_threads() or 4)))
    torch.manual_seed(int(cfg["seed"]))
    tr = cfg["surrogate"]["train"]
    steps_cap = int(max_steps or tr["max_steps"])

    splits = split_world_ids(cfg)
    print(f"worlds: train={len(splits['train'])} val={len(splits['val'])} "
          f"holdout-regime={len(splits['holdout'])}")
    train_recs = load_records(cfg, splits["train"])
    val_recs = load_records(cfg, splits["val"])
    crop = int(tr["crop"])
    train_ds = FirePairDataset(train_recs, crop, rng_seed=int(cfg["seed"]) + 1,
                               oversample_early=int(tr.get("oversample_early", 1)))
    val_ds = FirePairDataset(val_recs, crop, rng_seed=int(cfg["seed"]) + 2)
    print(f"pairs: train={len(train_ds)} val={len(val_ds)}")

    g = torch.Generator().manual_seed(int(cfg["seed"]))
    train_dl = DataLoader(train_ds, batch_size=int(tr["batch_size"]), shuffle=True,
                          generator=g, num_workers=0, drop_last=True)
    val_dl = DataLoader(val_ds, batch_size=int(tr["batch_size"]), num_workers=0)

    model = FireUNet(int(cfg["surrogate"]["model"]["base_channels"]))
    n_params = sum(p.numel() for p in model.parameters())
    print(f"model params: {n_params:,}")
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]))

    d = ckpt_dir(cfg)
    latest, best = d / "latest.pt", d / "best.pt"
    step, best_iou = 0, -1.0
    if resume and latest.exists():
        state = torch.load(latest, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        step, best_iou = state["step"], state["best_iou"]
        torch.set_rng_state(state["torch_rng"])
        print(f"resumed from step {step} (best val IoU {best_iou:.4f})")

    model.train()
    t0 = time.perf_counter()
    patience_left = int(tr["early_stop_patience"])
    running = []
    stop = False
    while step < steps_cap and not stop:
        for x, y in train_dl:
            logits = model(x)
            loss = loss_fn(logits, y)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running.append(loss.item())
            step += 1

            if step % int(tr["val_every"]) == 0:
                val_iou = validate(model, val_dl)
                dt = time.perf_counter() - t0
                print(f"step {step:5d}  loss {np.mean(running):.4f}  "
                      f"val IoU@+10 {val_iou:.4f}  [{dt/60:.1f} min]", flush=True)
                running = []
                if val_iou > best_iou:
                    best_iou = val_iou
                    patience_left = int(tr["early_stop_patience"])
                    save_ckpt(best, model, opt, step, best_iou)
                else:
                    patience_left -= 1
                    if patience_left <= 0:
                        print("early stopping: no val improvement")
                        stop = True
            if step % int(tr["checkpoint_every"]) == 0:
                save_ckpt(latest, model, opt, step, best_iou)
            if step >= steps_cap:
                stop = True
            if max_minutes and (time.perf_counter() - t0) / 60 > max_minutes:
                print(f"time budget {max_minutes} min reached; stopping at best checkpoint")
                stop = True
            if stop:
                break

    save_ckpt(latest, model, opt, step, best_iou)
    if not best.exists():  # no val improvement ever recorded (tiny smoke runs)
        save_ckpt(best, model, opt, step, best_iou)
    print(f"done at step {step}; best val IoU@+10 = {best_iou:.4f} -> {best}")
    return best


def main() -> None:
    ap = argparse.ArgumentParser(description="Train the Emberline fire surrogate")
    ap.add_argument("--no-resume", action="store_true")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--max-minutes", type=float, default=None,
                    help="hard wall-clock cap; stops gracefully at best checkpoint")
    args = ap.parse_args()
    train(load_config(), resume=not args.no_resume, max_steps=args.max_steps,
          max_minutes=args.max_minutes)


if __name__ == "__main__":
    main()
