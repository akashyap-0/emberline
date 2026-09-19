"""Train the real-data smoke detectors FROM SCRATCH. Real data only.

    python -m emberline.detect.train_real [--skip-cnn] [--epochs N]

Two models, both starting from random/none — no synthetic-trained weights
are loaded anywhere in this module:

* GBM (primary): sklearn HistGradientBoostingClassifier on the window
  features of emberline/detect/features_real.py. Train split only.
* 1D-CNN (comparison): the SmokeCNN architecture re-instantiated from
  scratch with the real retained-channel count, trained on raw normalized
  windows. Early stopping on the val split; checkpoint-resume supported.

The decision threshold is selected on TRAIN/VAL ONLY (Youden's J on the
val split at deployment stride); the test sessions are never touched here.

Artifacts (committed, small):
  data/checkpoints/detect_real_gbm.pkl   — model + threshold + metadata
  data/checkpoints/detect_real_cnn.pt    — weights + norm stats + threshold
"""

from __future__ import annotations

import argparse
import pickle
import sys
import time

import numpy as np
import torch
import torch.nn as nn

from emberline.data import REPO_ROOT
from emberline.detect.features_real import (
    EVAL_STRIDE,
    FEATURE_NAMES,
    RETAINED_CHANNELS,
    TRAIN_STRIDE,
    split_windows,
    window_features,
)
from emberline.detect.model import SmokeCNN

CKPT_DIR = REPO_ROOT / "data" / "checkpoints"
GBM_PATH = CKPT_DIR / "detect_real_gbm.pkl"
CNN_PATH = CKPT_DIR / "detect_real_cnn.pt"
MAJORITY_BASELINE = 0.715  # dataset-wide positive rate; printed beside scores


def pick_threshold(y: np.ndarray, p: np.ndarray) -> tuple[float, float]:
    """Threshold by Youden's J (TPR - FPR) on (y, p) — VAL data only.

    Not F1-max: the val session is 87.5% positive, so the F1-maximizing
    threshold degenerates to ~0 ("alert always", F1 0.933 with zero
    specificity) — useless at deployment where positives are rare and
    false alarms are the cost. Youden's J is prior-free, so the choice
    survives the prior mismatch. Reported beside it: F1 at this threshold.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(y, p)
    i = int(np.argmax(tpr - fpr))
    t = float(thr[i])
    pred = p >= t
    tp = float((pred & (y == 1)).sum())
    prec = tp / max(pred.sum(), 1)
    rec = tp / max((y == 1).sum(), 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)
    return t, float(f1)


def train_gbm(train, val) -> dict:
    from sklearn.ensemble import HistGradientBoostingClassifier

    Xtr, ytr = window_features(train), train["y"]
    Xva, yva = window_features(val), val["y"]
    t0 = time.time()
    # Hyperparameters selected on the VAL split only (grid scored by val
    # ROC-AUC; test sessions untouched). The decisive one is
    # max_features=0.5: with only 2 train sessions, ambient level features
    # (temperature/pressure means) separate train perfectly but FLIP
    # direction on the val session (measured: Temperature[C]:mean AUC 0.027
    # on train vs 0.990 on val). Column subsampling stops every tree from
    # locking onto those session-specific levels and moves the ensemble to
    # the cross-session-consistent particulate channels. Without it the val
    # ROC-AUC was 0.24; with it, 0.996.
    gbm = HistGradientBoostingClassifier(
        max_iter=300, learning_rate=0.1, max_depth=3,
        l2_regularization=10.0, max_features=0.5,
        early_stopping=False, random_state=1337,
    )
    gbm.fit(Xtr, ytr)
    pva = gbm.predict_proba(Xva)[:, 1]
    thr, f1 = pick_threshold(yva, pva)
    wall = time.time() - t0
    print(f"GBM: {gbm.n_iter_} iters, {wall:.1f}s; val F1@thr={f1:.4f} (thr={thr:.4f}) "
          f"[majority baseline acc {MAJORITY_BASELINE:.1%}]")
    return {
        "model": gbm,
        "threshold": thr,
        "val_f1": f1,
        "feature_names": FEATURE_NAMES,
        "train_stride": TRAIN_STRIDE,
        "provenance": "trained from scratch on real data only (kaggle_smoke train sessions; "
                      "no synthetic samples, no synthetic-pretrained init)",
    }


def _norm_stats(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-channel mean/std over TRAIN windows only."""
    mu = x.mean(axis=(0, 2), keepdims=True)
    sd = x.std(axis=(0, 2), keepdims=True) + 1e-6
    return mu.astype(np.float32), sd.astype(np.float32)


def train_cnn(train, val, epochs: int, resume: bool) -> dict:
    torch.manual_seed(1337)
    np.random.seed(1337)
    mu, sd = _norm_stats(train["x"])
    Xtr = torch.from_numpy((train["x"] - mu) / sd)
    ytr = torch.from_numpy(train["y"].astype(np.float32))
    Xva = torch.from_numpy((val["x"] - mu) / sd)
    yva_np = val["y"].astype(np.float32)

    model = SmokeCNN(in_channels=len(RETAINED_CHANNELS))
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    start_ep, best_val, best_state, patience_left = 0, -1.0, None, 6

    if resume and CNN_PATH.exists():
        ck = torch.load(CNN_PATH, map_location="cpu", weights_only=False)
        if ck.get("in_channels") == len(RETAINED_CHANNELS) and not ck.get("finished"):
            model.load_state_dict(ck["state_dict"])
            opt.load_state_dict(ck["optimizer"])
            start_ep = ck["epoch"] + 1
            best_val = ck["best_val_ap"]
            best_state = ck.get("best_state_dict", None)
            patience_left = ck.get("patience_left", 6)
            print(f"resuming CNN at epoch {start_ep} (best val AP {best_val:.4f})")

    from sklearn.metrics import average_precision_score

    bs = 256
    n = len(Xtr)
    for ep in range(start_ep, epochs):
        model.train()
        perm = torch.randperm(n, generator=torch.Generator().manual_seed(1337 + ep))
        tot = 0.0
        for i in range(0, n, bs):
            j = perm[i : i + bs]
            logits = model(Xtr[j])
            loss = nn.functional.binary_cross_entropy_with_logits(logits, ytr[j])
            opt.zero_grad(); loss.backward(); opt.step()
            tot += float(loss) * len(j)
        model.eval()
        with torch.no_grad():
            pva = torch.sigmoid(model(Xva)).numpy()
        ap = float(average_precision_score(yva_np, pva))
        improved = ap > best_val + 1e-4
        if improved:
            best_val, patience_left = ap, 6
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience_left -= 1
        print(f"epoch {ep:02d}: train loss {tot/n:.4f}, val PR-AUC {ap:.4f}"
              + (" *" if improved else ""))
        torch.save(
            {
                "state_dict": model.state_dict(), "optimizer": opt.state_dict(),
                "epoch": ep, "best_val_ap": best_val, "best_state_dict": best_state,
                "patience_left": patience_left, "in_channels": len(RETAINED_CHANNELS),
                "norm_mu": mu, "norm_sd": sd, "finished": False,
            },
            CNN_PATH,
        )
        if patience_left <= 0:
            print("early stop")
            break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        pva = torch.sigmoid(model(Xva)).numpy()
    thr, f1 = pick_threshold(yva_np, pva)
    print(f"CNN: best val PR-AUC {best_val:.4f}; val F1@thr={f1:.4f} (thr={thr:.4f})")
    return {
        "state_dict": best_state, "in_channels": len(RETAINED_CHANNELS),
        "channels": list(RETAINED_CHANNELS), "norm_mu": mu, "norm_sd": sd,
        "threshold": thr, "val_f1": f1, "best_val_ap": best_val, "finished": True,
        "provenance": "trained from scratch on real data only (kaggle_smoke train sessions; "
                      "random init, no synthetic pretraining)",
    }


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-cnn", action="store_true")
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args(argv)

    CKPT_DIR.mkdir(parents=True, exist_ok=True)
    print("loading windows (train stride "
          f"{TRAIN_STRIDE}s, val stride {EVAL_STRIDE}s)...")
    train = split_windows("train", TRAIN_STRIDE)
    val = split_windows("val", EVAL_STRIDE)
    print(f"train: {len(train['y'])} windows ({train['y'].mean():.1%} pos), "
          f"val: {len(val['y'])} windows ({val['y'].mean():.1%} pos)")

    art = train_gbm(train, val)
    with open(GBM_PATH, "wb") as f:
        pickle.dump(art, f)
    print(f"saved {GBM_PATH}")

    if not args.skip_cnn:
        ck = train_cnn(train, val, args.epochs, resume=not args.no_resume)
        torch.save(ck, CNN_PATH)
        print(f"saved {CNN_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
