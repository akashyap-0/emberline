"""Train the NDWS spread UNet FROM SCRATCH on the train shards. Real data only.

    python -m emberline.spread.train_real [--minutes 110] [--epochs 60]

* Data: the dataset's own shipped split — 15 train shards for gradient
  steps, 2 eval shards for early stopping and threshold choice. The 2 test
  shards are never read here.
* Class imbalance: burned pixels are ~1% of labeled pixels; the loss is
  BCE-with-logits with pos_weight = (labeled negatives / labeled positives)
  computed on the train shards.
* Uncertain pixels: FireMask == -1 per the dataset spec — EXCLUDED from the
  loss via masking (and from every metric downstream). PrevFireMask == -1 in
  the input is left as its raw value for the network to see, standard NDWS
  practice.
* Normalization: per-channel mean/std over the 11 covariates, computed on
  TRAIN shards only, stored in the checkpoint. The PrevFireMask channel is
  passed raw (-1/0/1).
* Budget: wall-clock cap (default 110 min, hard cap 2 h per the plan) with
  per-epoch checkpointing to data/checkpoints/spread_real_unet.pt; rerunning
  the command resumes from the last epoch. Early stopping: eval PR-AUC,
  patience 4.
"""

from __future__ import annotations

import argparse
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from emberline.data import REPO_ROOT
from emberline.data.ndws import CHANNELS, load_split
from emberline.spread.model import SpreadUNet, param_count

CKPT = REPO_ROOT / "data" / "checkpoints" / "spread_real_unet.pt"
N_COV = len(CHANNELS) - 1  # 11 covariates; channel 11 is PrevFireMask


def train_norm_stats(xs: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Per-covariate mean/std over train shards (sampled rows for speed)."""
    acc = []
    for x in xs:
        step = max(1, len(x) // 200)
        acc.append(np.asarray(x[::step, :N_COV], dtype=np.float32))
    a = np.concatenate(acc)
    mu = a.mean(axis=(0, 2, 3))
    sd = a.std(axis=(0, 2, 3)) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def batch_tensor(xs, idx_pairs, mu, sd) -> torch.Tensor:
    """Assemble a standardized float32 batch from mmap'd shards."""
    rows = np.stack([np.asarray(xs[s][i], dtype=np.float32) for s, i in idx_pairs])
    rows[:, :N_COV] = (rows[:, :N_COV] - mu[None, :, None, None]) / sd[None, :, None, None]
    return torch.from_numpy(rows)


def eval_pr_auc(model, xs, ys, mu, sd, batch: int = 64) -> float:
    from sklearn.metrics import average_precision_score

    model.eval()
    probs, labels = [], []
    with torch.no_grad():
        for x, y in zip(xs, ys):
            for i in range(0, len(x), batch):
                xb = np.asarray(x[i:i + batch], dtype=np.float32)
                xb[:, :N_COV] = (xb[:, :N_COV] - mu[None, :, None, None]) / sd[None, :, None, None]
                p = torch.sigmoid(model(torch.from_numpy(xb))).numpy()
                yb = np.asarray(y[i:i + batch])
                m = yb != -1
                probs.append(p[m]); labels.append(yb[m])
    return float(average_precision_score(np.concatenate(labels), np.concatenate(probs)))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=110.0)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    torch.manual_seed(1337)
    rng = np.random.default_rng(1337)
    t_start = time.time()

    xtr, ytr = load_split("train")
    xev, yev = load_split("val")
    n_tiles = sum(len(x) for x in xtr)
    print(f"train tiles: {n_tiles} over {len(xtr)} shards; eval tiles: "
          f"{sum(len(x) for x in xev)}")

    # pos_weight and norm stats from TRAIN only
    pos = neg = 0
    for y in ytr:
        yy = np.asarray(y)
        pos += int((yy == 1).sum()); neg += int((yy == 0).sum())
    pos_weight = neg / max(pos, 1)
    print(f"train burned-pixel rate {pos/(pos+neg):.4%}; pos_weight {pos_weight:.1f}")
    mu, sd = train_norm_stats(xtr)

    model = SpreadUNet(in_channels=len(CHANNELS))
    n_par = param_count(model)
    assert n_par <= 5_000_000, f"UNet has {n_par} params, cap is 5M"
    print(f"UNet params: {n_par:,}")
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    start_ep, best_ap, best_state, patience = 0, -1.0, None, 4

    if not args.no_resume and CKPT.exists():
        ck = torch.load(CKPT, map_location="cpu", weights_only=False)
        if not ck.get("finished"):
            model.load_state_dict(ck["state_dict"])
            opt.load_state_dict(ck["optimizer"])
            start_ep = ck["epoch"] + 1
            best_ap = ck["best_eval_ap"]
            best_state = ck.get("best_state_dict")
            patience = ck.get("patience_left", 4)
            print(f"resuming at epoch {start_ep} (best eval PR-AUC {best_ap:.4f})")

    pairs = [(s, i) for s, x in enumerate(xtr) for i in range(len(x))]
    pw = torch.tensor(pos_weight, dtype=torch.float32)
    deadline = t_start + args.minutes * 60

    for ep in range(start_ep, args.epochs):
        model.train()
        order = rng.permutation(len(pairs))
        tot, seen = 0.0, 0
        t_ep = time.time()
        for k in range(0, len(order), args.batch):
            sel = [pairs[j] for j in order[k:k + args.batch]]
            xb = batch_tensor(xtr, sel, mu, sd)
            yb = torch.from_numpy(np.stack(
                [np.asarray(ytr[s][i]) for s, i in sel]).astype(np.float32))
            mask = yb != -1
            logits = model(xb)
            loss = nn.functional.binary_cross_entropy_with_logits(
                logits[mask], yb[mask], pos_weight=pw)
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * int(mask.sum()); seen += int(mask.sum())
            if time.time() > deadline:
                print("wall-clock cap hit mid-epoch; checkpointing and stopping")
                break
        ap_ev = eval_pr_auc(model, xev, yev, mu, sd)
        improved = ap_ev > best_ap + 1e-4
        if improved:
            best_ap, patience = ap_ev, 4
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
        print(f"epoch {ep:02d}: loss {tot/max(seen,1):.4f}, eval PR-AUC {ap_ev:.4f}"
              f"{' *' if improved else ''} ({time.time()-t_ep:.0f}s)")
        torch.save({
            "state_dict": model.state_dict(), "optimizer": opt.state_dict(),
            "epoch": ep, "best_eval_ap": best_ap, "best_state_dict": best_state,
            "patience_left": patience, "norm_mu": mu, "norm_sd": sd,
            "pos_weight": pos_weight, "channels": list(CHANNELS),
            "params": n_par, "finished": False,
            "provenance": "trained from scratch on real NDWS train shards only; "
                          "no synthetic data, no pretrained weights",
        }, CKPT)
        if patience <= 0:
            print("early stop")
            break
        if time.time() > deadline:
            break

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    ck["state_dict"] = best_state or ck["state_dict"]
    ck["finished"] = True
    torch.save(ck, CKPT)
    print(f"saved {CKPT} (best eval PR-AUC {best_ap:.4f}, "
          f"wall {time.time()-t_start:.0f}s)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
