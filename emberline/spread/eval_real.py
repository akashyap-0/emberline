"""Evaluate NDWS spread models on the TEST shards only. Real data only.

    python -m emberline.spread.eval_real

Models compared:
* persistence baseline — next-day fire mask = today's fire mask
  (PrevFireMask == 1; uncertain prev pixels predict "no fire"). The floor
  any learned model must beat.
* SpreadUNet — trained from scratch on the train shards; its binary
  threshold is chosen on the EVAL shards (max IoU), never on test.

Pixels with FireMask == -1 (uncertain, per the dataset spec) are excluded
from every metric. Metrics are pooled over all labeled test pixels: IoU,
precision, recall of burned cells at +1 day, plus PR-AUC for the UNet
(persistence is binary — no score to sweep).

Regime note: 1 km / daily satellite-scale spread — a different regime from
the 10 m / minute-scale simulator elsewhere in this repo; these numbers are
the real-data reference point, not a claim about the node-scale system.

Writes metrics/spread_real.json and demo/out/real/spread_real_tiles.png.
"""

from __future__ import annotations

import json
import sys

import numpy as np
import torch

from emberline.data import REPO_ROOT
from emberline.data.ndws import CHANNELS, load_split
from emberline.spread.model import SpreadUNet
from emberline.spread.train_real import CKPT, N_COV

METRICS_PATH = REPO_ROOT / "metrics" / "spread_real.json"
OUT_DIR = REPO_ROOT / "demo" / "out" / "real"


def collect_probs(model, xs, ys, mu, sd, batch: int = 64):
    """Pooled (labels, probs, prev_mask) over labeled pixels of a split."""
    probs, labels, prevs = [], [], []
    model.eval()
    with torch.no_grad():
        for x, y in zip(xs, ys):
            for i in range(0, len(x), batch):
                xb = np.asarray(x[i:i + batch], dtype=np.float32)
                prev = xb[:, N_COV].copy()
                xb[:, :N_COV] = (xb[:, :N_COV] - mu[None, :, None, None]) / sd[None, :, None, None]
                p = torch.sigmoid(model(torch.from_numpy(xb))).numpy()
                yb = np.asarray(y[i:i + batch])
                m = yb != -1
                probs.append(p[m]); labels.append(yb[m]); prevs.append(prev[m])
    return (np.concatenate(labels).astype(np.int8),
            np.concatenate(probs),
            np.concatenate(prevs))


def binary_metrics(y: np.ndarray, pred: np.ndarray) -> dict:
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    iou = tp / max(tp + fp + fn, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    return {"iou": round(iou, 4), "precision": round(prec, 4),
            "recall": round(rec, 4), "tp": tp, "fp": fp, "fn": fn}


def pick_iou_threshold(y: np.ndarray, p: np.ndarray) -> float:
    """Threshold maximizing pooled IoU — computed on the EVAL split only."""
    best_t, best_iou = 0.5, -1.0
    for t in np.linspace(0.05, 0.95, 19):
        m = binary_metrics(y, (p >= t).astype(np.int8))
        if m["iou"] > best_iou:
            best_iou, best_t = m["iou"], float(t)
    return best_t


def tile_plot(model, mu, sd) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    xs, ys = load_split("test")
    x = np.asarray(xs[0][:400], dtype=np.float32)
    y = np.asarray(ys[0][:400])
    # pick tiles with substantial fire to show
    burn = (y == 1).sum(axis=(1, 2))
    idx = np.argsort(burn)[-4:]
    xb = x[idx].copy()
    prev = xb[:, N_COV].copy()
    xb[:, :N_COV] = (xb[:, :N_COV] - mu[None, :, None, None]) / sd[None, :, None, None]
    with torch.no_grad():
        p = torch.sigmoid(model(torch.from_numpy(xb))).numpy()
    fig, axes = plt.subplots(len(idx), 4, figsize=(10, 2.6 * len(idx)))
    for r, i in enumerate(idx):
        panels = [(prev[r], "PrevFireMask (t)"), (y[i], "FireMask (t+1)"),
                  ((prev[r] == 1).astype(float), "persistence"), (p[r], "UNet P(fire)")]
        for c, (img, title) in enumerate(panels):
            ax = axes[r, c]
            ax.imshow(img, cmap="inferno", vmin=-1 if c < 2 else 0, vmax=1)
            ax.set_xticks([]); ax.set_yticks([])
            if r == 0:
                ax.set_title(title, fontsize=9)
    fig.suptitle("NDWS test tiles — 1 km/daily scale (real data)", fontsize=11)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / "spread_real_tiles.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    from sklearn.metrics import average_precision_score

    ck = torch.load(CKPT, map_location="cpu", weights_only=False)
    model = SpreadUNet(in_channels=len(CHANNELS))
    model.load_state_dict(ck["state_dict"])
    mu, sd = ck["norm_mu"], ck["norm_sd"]

    # threshold on EVAL shards
    y_ev, p_ev, _ = collect_probs(model, *load_split("val"), mu, sd)
    thr = pick_iou_threshold(y_ev, p_ev)
    print(f"threshold {thr:.2f} (max-IoU on eval shards; eval PR-AUC "
          f"{average_precision_score(y_ev, p_ev):.4f})")

    y, p, prev = collect_probs(model, *load_split("test"), mu, sd)
    print(f"test labeled pixels: {len(y):,} ({(y == 1).mean():.3%} burned)")

    pers = binary_metrics(y, (prev == 1).astype(np.int8))
    unet = binary_metrics(y, (p >= thr).astype(np.int8))
    unet["pr_auc"] = round(float(average_precision_score(y, p)), 4)
    unet["threshold"] = thr

    print(f"persistence: IoU {pers['iou']:.4f} | precision {pers['precision']:.4f} "
          f"| recall {pers['recall']:.4f}")
    print(f"UNet:        IoU {unet['iou']:.4f} | precision {unet['precision']:.4f} "
          f"| recall {unet['recall']:.4f} | PR-AUC {unet['pr_auc']:.4f} "
          f"(burned-pixel prevalence {(y == 1).mean():.4f})")
    verdict = "beats" if unet["iou"] > pers["iou"] else "DOES NOT BEAT"
    print(f"UNet {verdict} persistence on test IoU")

    results = {
        "note": ("Real data only; UNet trained from scratch on NDWS train shards, "
                 "threshold from eval shards, metrics on test shards. 1 km/daily "
                 "satellite regime — not the 10 m node-scale system."),
        "test_labeled_pixels": int(len(y)),
        "burned_prevalence": round(float((y == 1).mean()), 5),
        "persistence": pers,
        "unet": unet,
        "unet_params": int(ck.get("params", 0)),
        "best_eval_pr_auc": round(float(ck.get("best_eval_ap", -1)), 4),
    }
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    tile_plot(model, mu, sd)
    print(f"wrote {METRICS_PATH} and {OUT_DIR / 'spread_real_tiles.png'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
