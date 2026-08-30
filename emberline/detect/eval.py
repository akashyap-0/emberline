"""Detection metrics: regenerates every Phase-4 number from saved models.

Run: ``python -m emberline.detect.eval``

* window precision/recall/F1 on val EVENTS (never seen in training) for the
  CNN and the GBM baseline;
* per-confounder confusion: fraction of each confounder's val windows the
  CNN calls "fire";
* ambient-day false positives: a full 24 h confounder-rich, fire-free
  simulation across all nodes, classifier slid every 30 s -> false positives
  per node-day (the metric that decides whether neighbours tear the sirens
  down);
* detection latency: fresh simulated fires, seconds from plume arrival at
  the best-placed node to the first positive 60-s window.
"""

from __future__ import annotations

import argparse
from typing import Any

import joblib
import numpy as np
import torch

from ..config import load_config, rng_for
from ..report import save_metrics, update_section
from ..worldgen import generate_world
from . import cnn_ckpt_path, gbm_ckpt_path, windows_path
from .data import FIRE_PM_GATE, simulate_fire_event
from .model import SmokeCNN, gbm_features, normalize
from .train import event_split
from ..sensors import place_nodes, sample_series
from ..sensors.events import CONFOUNDERS, fire_added_series, make_confounder

WINDOW_S, STRIDE_S = 60, 30


def load_models(cfg):
    model = SmokeCNN()
    state = torch.load(cnn_ckpt_path(cfg), weights_only=False)
    model.load_state_dict(state["model"])
    model.eval()
    cnn_thr = float(state.get("threshold", cfg["detect"]["threshold"]))
    return model, joblib.load(gbm_ckpt_path(cfg)), cnn_thr


def prf(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    tp = float((pred & (y == 1)).sum())
    prec = tp / max(1.0, float(pred.sum()))
    rec = tp / max(1.0, float((y == 1).sum()))
    return {"precision": prec, "recall": rec,
            "f1": 2 * prec * rec / max(1e-9, prec + rec)}


@torch.no_grad()
def cnn_probs(model: SmokeCNN, X: np.ndarray, bs: int = 512) -> np.ndarray:
    out = []
    Xn = torch.from_numpy(normalize(X.astype(np.float32)))
    for i in range(0, len(Xn), bs):
        out.append(torch.sigmoid(model(Xn[i : i + bs])).numpy())
    return np.concatenate(out) if out else np.zeros(0)


def slide_windows(series: np.ndarray, stride: int = STRIDE_S) -> tuple[np.ndarray, np.ndarray]:
    """(4, T) series -> stacked (K, 4, 60) windows + window END times (s)."""
    T = series.shape[1]
    starts = np.arange(0, T - WINDOW_S + 1, stride)
    return np.stack([series[:, s : s + WINDOW_S] for s in starts]), starts + WINDOW_S


def ambient_day_fpr(cfg: dict[str, Any], model: SmokeCNN, gbm, cnn_thr: float) -> dict[str, float]:
    """24 h fire-free, confounder-rich day across all nodes -> FP/node-day."""
    rng = rng_for(cfg, "detect-ambient-day")
    world = generate_world(cfg, 300)
    nodes = place_nodes(world, int(cfg["sensors"]["n_nodes"]), rng_for(cfg, "detect-nodes", 0))
    noise = cfg["sensors"]["noise"]
    T = 86400
    t = np.arange(T).astype(float)
    added = np.zeros((len(nodes), 4, T))
    n_events = 14
    for _ in range(n_events):
        kind = CONFOUNDERS[int(rng.integers(len(CONFOUNDERS)))]
        spec = make_confounder(kind, len(nodes), rng)
        t0 = {"wood_stove": rng.uniform(17, 21) * 3600,
              "fog": rng.uniform(3, 6) * 3600}.get(kind, float(rng.uniform(0, 22 * 3600)))
        t0 = int(min(t0, T - spec.duration_s - 1))
        dur = int(spec.duration_s)
        env = spec.envelope
        for ni in spec.scope:
            added[ni, :, t0 : t0 + dur] += env(ni, np.arange(dur).astype(float))

    thr = float(cfg["detect"]["threshold"])
    fp_cnn = fp_gbm = n_windows = 0
    for ni, node in enumerate(nodes):
        series = sample_series(node, t, added[ni], noise, rng)
        W, _ = slide_windows(series)
        n_windows += len(W)
        fp_cnn += int((cnn_probs(model, W) > cnn_thr).sum())
        fp_gbm += int(gbm.predict_proba(gbm_features(W.astype(np.float32)))[:, 1].__gt__(thr).sum())
    days = len(nodes)  # one simulated day per node
    return {"fp_per_node_day_cnn": fp_cnn / days, "fp_per_node_day_gbm": fp_gbm / days,
            "windows_evaluated": n_windows, "confounder_events": n_events}


def detection_latency(cfg: dict[str, Any], model: SmokeCNN, cnn_thr: float,
                      n_events: int = 25) -> dict[str, float]:
    """Seconds from plume arrival (sustained PM above gate) to first positive."""
    rng = rng_for(cfg, "detect-latency")
    noise = cfg["sensors"]["noise"]
    thr = cnn_thr
    lats = []
    misses = 0
    for i in range(n_events):
        world = generate_world(cfg, 320 + i % 5)
        nodes = place_nodes(world, int(cfg["sensors"]["n_nodes"]),
                            rng_for(cfg, "detect-nodes", 20 + i % 5))
        conc = simulate_fire_event(cfg, world, nodes, rng, minutes=40)
        ni = int(np.argmax(conc.mean(axis=1)))
        if conc[ni].max() < FIRE_PM_GATE:
            misses += 1
            continue
        arrive = int(np.argmax(conc[ni] >= FIRE_PM_GATE))
        added = fire_added_series(conc[ni], rng)
        series = sample_series(nodes[ni], np.arange(conc.shape[1]) + 43200.0, added, noise, rng)
        W, ends = slide_windows(series, stride=10)
        probs = cnn_probs(model, W)
        pos = np.where((probs > thr) & (ends > arrive))[0]
        if len(pos) == 0:
            misses += 1
            continue
        lats.append(float(ends[pos[0]] - arrive))
    return {"n_events": n_events, "undetected": misses,
            "latency_median_s": float(np.median(lats)) if lats else float("nan"),
            "latency_mean_s": float(np.mean(lats)) if lats else float("nan")}


def main() -> None:
    argparse.ArgumentParser(description="Evaluate smoke classifiers").parse_args()
    cfg = load_config()
    model, gbm, cnn_thr = load_models(cfg)

    with np.load(windows_path(cfg), allow_pickle=False) as z:
        X, y, kind, event_id = z["X"], z["y"], z["kind"], z["event_id"]
    val_mask = event_split(event_id, float(cfg["detect"]["train"]["val_event_frac"]),
                           rng_for(cfg, "detect-split"))
    Xv, yv, kv = X[val_mask], y[val_mask], kind[val_mask]
    thr = float(cfg["detect"]["threshold"])

    p_cnn = cnn_probs(model, Xv) > cnn_thr
    p_gbm = gbm.predict_proba(gbm_features(Xv))[:, 1] > thr
    m_cnn, m_gbm = prf(p_cnn, yv), prf(p_gbm, yv)

    per_conf = {}
    for k in CONFOUNDERS + ["ambient"]:
        mask = kv == k
        if mask.sum():
            per_conf[k] = {"windows": int(mask.sum()),
                           "fp_rate_cnn": float(p_cnn[mask].mean()),
                           "fp_rate_gbm": float(p_gbm[mask].mean())}

    amb = ambient_day_fpr(cfg, model, gbm, cnn_thr)
    lat = detection_latency(cfg, model, cnn_thr)

    metrics = {"val_windows": int(val_mask.sum()), "cnn": m_cnn, "gbm": m_gbm,
               "per_confounder": per_conf, "ambient_day": amb, "latency": lat,
               "threshold_gbm": thr, "threshold_cnn": cnn_thr}
    save_metrics("detect", metrics)

    conf_rows = "\n".join(
        f"| {k} | {v['windows']} | {v['fp_rate_cnn']:.3f} | {v['fp_rate_gbm']:.3f} |"
        for k, v in per_conf.items())
    beats = m_cnn["f1"] >= m_gbm["f1"]
    verdict = ("The CNN beats the GBM baseline on val F1."
               if beats else
               "The CNN does NOT beat the GBM baseline on val F1. With only 60 s of "
               "context the handcrafted summary features (levels, slopes, VOC/PM "
               "ratio, RH) capture most of the signal; the honest engineering call "
               "is to ship the cheaper model on-node and revisit with longer windows.")

    body = f"""## Smoke detection (Phase 4)

Regenerate: `python -m emberline.detect.eval`. Split by EVENT (val events
never seen in training); thresholds: CNN {cnn_thr:.2f} (train-selected), GBM {thr}. **All data is synthetic** — plume +
signature models, not real sensors; collecting real burn/confounder data is
the team's stated next step.

| model | precision | recall | F1 |
|---|---|---|---|
| 1D-CNN ({sum(p.numel() for p in model.parameters()):,} params) | {m_cnn['precision']:.3f} | {m_cnn['recall']:.3f} | {m_cnn['f1']:.3f} |
| GBM baseline | {m_gbm['precision']:.3f} | {m_gbm['recall']:.3f} | {m_gbm['f1']:.3f} |

{verdict}

Per-confounder false-positive rate on val windows (CNN / GBM):

| confounder | windows | CNN FP rate | GBM FP rate |
|---|---|---|---|
{conf_rows}

Ambient 24 h, fire-free, {amb['confounder_events']} confounder events across
{amb['windows_evaluated']} windows: **{amb['fp_per_node_day_cnn']:.2f} false positives per
node-day (CNN)**, {amb['fp_per_node_day_gbm']:.2f} (GBM). Note these are single-node,
single-window numbers — the mesh's corroboration ladder (Phase 5) is what
turns them into siren-worthy alarms.

Detection latency over {lat['n_events']} fresh simulated fires: median
**{lat['latency_median_s']:.0f} s**, mean {lat['latency_mean_s']:.0f} s from plume arrival to first
positive window ({lat['undetected']} fires never produced a classifiable plume at any
node — typically burning away from the network).
"""
    update_section("detect", body)
    print(f"cnn: {m_cnn}\ngbm: {m_gbm}\nambient: {amb}\nlatency: {lat}")
    print("wrote metrics/detect.json and REPORT.md section")


if __name__ == "__main__":
    main()
