"""Evaluate the real-data smoke detectors on the manifest TEST sessions only.

    python -m emberline.detect.eval_real

Test sessions (fixed in data/real/MANIFEST.yaml, never tuned on):
  session 2 — 1,154 s fire session (97.1% positive): detection latency.
  session 4 — 5,744 s fire-free session: false positives per node-day.

Reported per model: precision / recall / F1 at the val-chosen threshold,
PR-AUC, ROC-AUC, confusion matrix — each beside the majority-class
baseline (71.5% positive across the dataset; predicting "fire" always gets
71.5% accuracy and F1 0.834 dataset-wide, which is why accuracy is not the
headline metric). FP/node-day is computed over entirely fire-free windows;
detection latency is seconds from the first labeled-fire sample of a fire
session to the end of the first positive window at 1 s stride.

Writes metrics/detect_real.json and plots under demo/out/real/.
Real data only: models were trained from scratch on real windows; this
module loads no synthetic data and no synthetic-trained checkpoint.
"""

from __future__ import annotations

import json
import pickle
import sys

import numpy as np
import torch

from emberline.data import REPO_ROOT
from emberline.detect.features_real import (
    EVAL_STRIDE,
    split_windows,
    window_features,
)
from emberline.detect.model import SmokeCNN
from emberline.detect.train_real import CNN_PATH, GBM_PATH, MAJORITY_BASELINE

OUT_DIR = REPO_ROOT / "demo" / "out" / "real"
METRICS_PATH = REPO_ROOT / "metrics" / "detect_real.json"


def model_probs(test) -> dict[str, tuple[np.ndarray, float]]:
    """{model name: (P(fire) per window, val-chosen threshold)}."""
    out = {}
    with open(GBM_PATH, "rb") as f:
        art = pickle.load(f)
    out["gbm"] = (art["model"].predict_proba(window_features(test))[:, 1],
                  float(art["threshold"]))

    ck = torch.load(CNN_PATH, map_location="cpu", weights_only=False)
    model = SmokeCNN(in_channels=ck["in_channels"])
    model.load_state_dict(ck["state_dict"])
    model.eval()
    X = torch.from_numpy((test["x"] - ck["norm_mu"]) / ck["norm_sd"])
    with torch.no_grad():
        p = torch.cat([torch.sigmoid(model(X[i:i + 4096]))
                       for i in range(0, len(X), 4096)]).numpy()
    out["cnn"] = (p, float(ck["threshold"]))
    return out


def evaluate(test, p: np.ndarray, thr: float) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = test["y"].astype(int)
    pred = (p >= thr).astype(int)
    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 2 * prec * rec / max(prec + rec, 1e-12)

    # False positives per node-day over fire-free spans: predicted-positive
    # windows among windows containing no labeled-fire sample at all. At 1 s
    # stride each window is one decision second.
    clean = test["clean"]
    fp_clean = int((pred[clean] == 1).sum())
    clean_seconds = int(clean.sum())
    fp_per_day = fp_clean / max(clean_seconds, 1) * 86400.0

    # Detection latency per fire session: first labeled-fire row -> end of
    # first positive window at or after it.
    latencies = {}
    for s in np.unique(test["session"]):
        m = test["session"] == s
        ys, ps_, ends = y[m], pred[m], test["end"][m]
        fire_rows = np.nonzero(ys == 1)[0]
        if len(fire_rows) == 0:
            continue
        # first fire SAMPLE time: windows are labeled by last sample, and at
        # stride 1 window k ends at row end[k]; the first labeled-fire sample
        # in the session is at second end[fire_rows[0]].
        t0 = int(ends[fire_rows[0]])
        after = np.nonzero((ends >= t0) & (ps_ == 1))[0]
        latencies[int(s)] = (int(ends[after[0]]) - t0) if len(after) else None

    # Per-session breakdown. Session 4 is a temperature-altered duplicate of
    # TRAIN session 3 (manifest: duplicate_of), so session 2 is the only
    # genuinely held-out recording — its numbers are the honest headline.
    per_session = {}
    for s in np.unique(test["session"]):
        m = test["session"] == s
        ys, ps_ = y[m], pred[m]
        cl = clean & m
        per_session[int(s)] = {
            "windows": int(m.sum()),
            "positives": int(ys.sum()),
            "tp": int(((ps_ == 1) & (ys == 1)).sum()),
            "fp": int(((ps_ == 1) & (ys == 0)).sum()),
            "fn": int(((ps_ == 0) & (ys == 1)).sum()),
            "tn": int(((ps_ == 0) & (ys == 0)).sum()),
            "fp_per_node_day_fire_free": round(
                float((pred[cl] == 1).sum()) / max(int(cl.sum()), 1) * 86400.0, 2),
            "fire_free_seconds": int(cl.sum()),
        }

    return {
        "threshold": thr,
        "per_session": per_session,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "pr_auc": round(float(average_precision_score(y, p)), 4),
        "roc_auc": round(float(roc_auc_score(y, p)), 4),
        "confusion": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
        "fp_per_node_day_fire_free": round(fp_per_day, 2),
        "fire_free_seconds": clean_seconds,
        "detection_latency_s": latencies,
    }


def plots(test, probs: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import precision_recall_curve, roc_curve

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    y = test["y"].astype(int)

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
    for name, (p, thr) in probs.items():
        prec, rec, _ = precision_recall_curve(y, p)
        axes[0].plot(rec, prec, label=f"{name.upper()}")
        fpr, tpr, _ = roc_curve(y, p)
        axes[1].plot(fpr, tpr, label=f"{name.upper()}")
    axes[0].axhline(y.mean(), ls="--", c="gray",
                    label=f"prevalence {y.mean():.3f}")
    axes[0].set(xlabel="recall", ylabel="precision",
                title="PR — test sessions (real data)")
    axes[1].plot([0, 1], [0, 1], ls="--", c="gray", label="chance")
    axes[1].set(xlabel="FPR", ylabel="TPR", title="ROC — test sessions (real data)")
    for ax in axes:
        ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT_DIR / "detect_real_curves.png", dpi=130)
    plt.close(fig)

    fig, axes = plt.subplots(len(np.unique(test["session"])), 1,
                             figsize=(11, 6), sharex=False)
    for ax, s in zip(np.atleast_1d(axes), np.unique(test["session"])):
        m = test["session"] == s
        t = test["end"][m]
        ax.fill_between(t, 0, test["y"][m], color="tab:red", alpha=0.25,
                        step="mid", label="Fire Alarm label")
        for name, (p, thr) in probs.items():
            ax.plot(t, p[m], lw=0.8, label=f"{name.upper()} P(fire)")
        ax.set(ylabel=f"session {s}", ylim=(-0.02, 1.02))
        ax.grid(alpha=0.3)
    np.atleast_1d(axes)[0].legend(loc="center right", fontsize=8)
    np.atleast_1d(axes)[-1].set_xlabel("seconds since session start (1 Hz rows)")
    fig.suptitle("Model probability vs label — manifest test sessions")
    fig.tight_layout()
    fig.savefig(OUT_DIR / "detect_real_timeline.png", dpi=130)
    plt.close(fig)


def main(argv=None) -> int:
    test = split_windows("test", EVAL_STRIDE)
    print(f"test: {len(test['y'])} windows across sessions "
          f"{sorted(set(test['session'].tolist()))} "
          f"({test['y'].mean():.1%} positive; dataset majority baseline "
          f"{MAJORITY_BASELINE:.1%})")
    probs = model_probs(test)
    results = {"majority_baseline_accuracy": MAJORITY_BASELINE,
               "note": ("Real data only: trained from scratch on kaggle_smoke train "
                        "sessions; threshold from val session; test sessions 2 and 4 "
                        "only. 71.5% positive prior is nothing like deployment - "
                        "PR-AUC and FP/node-day are the informative numbers."),
               "models": {}}
    for name, (p, thr) in probs.items():
        r = evaluate(test, p, thr)
        results["models"][name] = r
        print(f"\n{name.upper()} (thr={thr:.4f}, chosen on val)")
        print(f"  precision {r['precision']:.4f} | recall {r['recall']:.4f} | "
              f"F1 {r['f1']:.4f}   [majority baseline: always-fire acc "
              f"{MAJORITY_BASELINE:.1%}]")
        print(f"  PR-AUC {r['pr_auc']:.4f} (prevalence {test['y'].mean():.3f}) | "
              f"ROC-AUC {r['roc_auc']:.4f} (chance 0.5)")
        print(f"  confusion tp={r['confusion']['tp']} fp={r['confusion']['fp']} "
              f"fn={r['confusion']['fn']} tn={r['confusion']['tn']}")
        print(f"  FP/node-day over fire-free spans: {r['fp_per_node_day_fire_free']} "
              f"({r['fire_free_seconds']} fire-free s)")
        print(f"  detection latency (s): {r['detection_latency_s']}")
        for s, ps in r["per_session"].items():
            print(f"    session {s}: tp={ps['tp']} fp={ps['fp']} fn={ps['fn']} "
                  f"tn={ps['tn']}, FP/day={ps['fp_per_node_day_fire_free']} "
                  f"({ps['fire_free_seconds']} fire-free s)")
    plots(test, probs)
    METRICS_PATH.parent.mkdir(exist_ok=True)
    METRICS_PATH.write_text(json.dumps(results, indent=2))
    print(f"\nwrote {METRICS_PATH} and plots in {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
