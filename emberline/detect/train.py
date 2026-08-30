"""Train the smoke CNN and the GBM baseline, split BY EVENT.

Event-level split: all windows of one event land entirely in train or
entirely in val (windows of one event are near-duplicates; a window split
would leak and flatter every metric). Ambient windows have their own event
ids and are split the same way.
"""

from __future__ import annotations

import argparse
from typing import Any

import joblib
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.ensemble import GradientBoostingClassifier

from ..config import load_config, rng_for
from . import cnn_ckpt_path, gbm_ckpt_path, windows_path
from .data import build_windows
from .model import SmokeCNN, gbm_features, normalize


def event_split(event_id: np.ndarray, val_frac: float, rng: np.random.Generator):
    """Boolean val mask: whole events assigned to val at random."""
    ids = np.unique(event_id)
    rng.shuffle(ids)
    n_val = max(1, int(len(ids) * val_frac))
    val_ids = set(ids[:n_val].tolist())
    return np.array([e in val_ids for e in event_id])


def train(cfg: dict[str, Any]) -> dict[str, float]:
    torch.manual_seed(int(cfg["seed"]))
    build_windows(cfg)
    with np.load(windows_path(cfg), allow_pickle=False) as z:
        X, y, event_id = z["X"], z["y"], z["event_id"]

    tr = cfg["detect"]["train"]
    val_mask = event_split(event_id, float(tr["val_event_frac"]), rng_for(cfg, "detect-split"))
    Xt, yt = X[~val_mask], y[~val_mask]
    Xv, yv = X[val_mask], y[val_mask]
    print(f"windows: train={len(yt)} ({yt.sum()} fire) val={len(yv)} ({yv.sum()} fire)")

    # --- GBM baseline -------------------------------------------------------
    gbm = GradientBoostingClassifier(random_state=int(cfg["seed"]), n_estimators=150,
                                     max_depth=3)
    gbm.fit(gbm_features(Xt), yt)
    gbm_val_acc = float((gbm.predict(gbm_features(Xv)) == yv).mean())
    joblib.dump(gbm, gbm_ckpt_path(cfg))

    # --- CNN ----------------------------------------------------------------
    model = SmokeCNN()
    print(f"cnn params: {sum(p.numel() for p in model.parameters()):,}")
    opt = torch.optim.Adam(model.parameters(), lr=float(tr["lr"]))
    Xt_t = torch.from_numpy(normalize(Xt))
    yt_t = torch.from_numpy(yt.astype(np.float32))
    Xv_t = torch.from_numpy(normalize(Xv))
    yv_t = torch.from_numpy(yv.astype(np.float32))
    pos_weight = torch.tensor((len(yt) - yt.sum()) / max(1, yt.sum()), dtype=torch.float32)

    bs = int(tr["batch_size"])
    best_f1, patience, best_state = -1.0, int(tr["early_stop_patience"]), None
    g = torch.Generator().manual_seed(int(cfg["seed"]))
    for epoch in range(int(tr["max_epochs"])):
        model.train()
        perm = torch.randperm(len(yt_t), generator=g)
        for i in range(0, len(perm) - bs + 1, bs):
            idx = perm[i : i + bs]
            logit = model(Xt_t[idx])
            loss = F.binary_cross_entropy_with_logits(logit, yt_t[idx], pos_weight=pos_weight)
            opt.zero_grad()
            loss.backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = (torch.sigmoid(model(Xv_t)) > float(cfg["detect"]["threshold"])).float()
        tp = float((pred * yv_t).sum())
        prec = tp / max(1.0, float(pred.sum()))
        rec = tp / max(1.0, float(yv_t.sum()))
        f1 = 2 * prec * rec / max(1e-9, prec + rec)
        print(f"epoch {epoch:2d}  val P {prec:.3f} R {rec:.3f} F1 {f1:.3f}", flush=True)
        if f1 > best_f1:
            best_f1, patience = f1, int(tr["early_stop_patience"])
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            patience -= 1
            if patience <= 0:
                print("early stopping")
                break

    assert best_state is not None
    # Operating threshold chosen on TRAIN windows (never val) to maximise F1:
    # pos_weight skews the raw 0.5 point toward recall; this recentres it.
    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        probs = torch.sigmoid(model(Xt_t)).numpy()
    best_thr, best_tf1 = 0.5, -1.0
    for thr in np.linspace(0.2, 0.95, 31):
        pred = probs > thr
        tp = float((pred & (yt == 1)).sum())
        p = tp / max(1.0, pred.sum())
        r = tp / max(1.0, (yt == 1).sum())
        f1 = 2 * p * r / max(1e-9, p + r)
        if f1 > best_tf1:
            best_thr, best_tf1 = float(thr), f1
    print(f"train-selected CNN threshold: {best_thr:.2f}")
    torch.save({"model": best_state, "val_f1": best_f1, "threshold": best_thr},
               cnn_ckpt_path(cfg))
    print(f"saved CNN (val F1 {best_f1:.3f}) and GBM (val acc {gbm_val_acc:.3f})")
    return {"cnn_val_f1": best_f1, "gbm_val_acc": gbm_val_acc}


def main() -> None:
    argparse.ArgumentParser(description="Train smoke classifiers").parse_args()
    train(load_config())


if __name__ == "__main__":
    main()
