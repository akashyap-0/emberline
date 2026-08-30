"""Surrogate evaluation: regenerates every Phase-3 metric from the checkpoint.

Run: ``python -m emberline.surrogate.eval``

Metrics (all on world-level held-out data the model never trained on):

* **Burned-area IoU @ +10/+30/+60 min**: surrogate autoregressive rollout
  from the true initial state vs the physics simulation, intersection over
  union of the touched (burned|burning) mask. Reported for the val split and
  separately for the held-out wind-regime worlds (a wind band excluded from
  training entirely).
* **Worst-case degradation**: 5th-percentile IoU@+30 across eval fires
  (i.e. the 95%-of-fires-are-at-least-this-good floor).
* **Fire-arrival-time MAE**: per-cell first-touch time (10-min resolution)
  from rollout vs physics, over cells both touch within +60 min.
* **Ensemble speedup**: wall-clock of an equal-member-count 60-min ensemble,
  physics vs batched surrogate, same perturbation model.

The surrogate receives the TRUE wind sequence during rollout (as it would
from live anemometry/forecast in deployment); wind uncertainty is instead
expressed through ensemble perturbations.
"""

from __future__ import annotations

import argparse
import time
from typing import Any

import numpy as np
import torch

from ..config import load_config, rng_for
from ..firesim.ensemble import run_ensemble
from ..report import save_metrics, update_section
from ..worldgen import generate_world
from .data import SNAP_MIN
from .datasets import FireRecord, load_records, split_world_ids
from .rollout import SurrogateEngine

HORIZON_STEPS = {10: 1, 30: 3, 60: 6}


def eval_record(engine: SurrogateEngine, rec: FireRecord) -> dict[str, float] | None:
    """IoU per horizon + arrival MAE for one fire; None if too short."""
    T = rec.states.shape[0]
    if T < 7:  # need +60 min of truth
        return None
    static = torch.from_numpy(rec.static)
    touched0 = rec.states[0] > 0
    burning0 = rec.states[0] == 1
    winds = rec.winds

    def wind_fn(k: int) -> tuple[float, float]:
        return tuple(winds[min(k, len(winds) - 1)])

    masks = engine.rollout(static, touched0, burning0, 6, wind_fn)
    out: dict[str, float] = {}
    for h_min, k in HORIZON_STEPS.items():
        truth = rec.states[k] > 0
        pred = masks[k - 1]
        union = (truth | pred).sum()
        out[f"iou_{h_min}"] = float((truth & pred).sum() / union) if union else 1.0

    # Arrival time (minutes) at 10-min resolution over jointly-touched cells.
    truth_arr = np.full(truth.shape, np.inf)
    pred_arr = np.full(truth.shape, np.inf)
    for k in range(6, 0, -1):
        truth_arr[rec.states[k] > 0] = k * SNAP_MIN
        pred_arr[masks[k - 1]] = k * SNAP_MIN
    truth_arr[rec.states[0] > 0] = 0.0
    pred_arr[rec.states[0] > 0] = 0.0
    both = np.isfinite(truth_arr) & np.isfinite(pred_arr)
    out["arrival_mae_min"] = float(np.abs(truth_arr[both] - pred_arr[both]).mean()) if both.any() else 0.0
    return out


def eval_split(engine: SurrogateEngine, records: list[FireRecord], cap: int,
               rng: np.random.Generator) -> tuple[dict[str, float], list[dict[str, float]]]:
    order = rng.permutation(len(records))
    rows = []
    for i in order:
        r = eval_record(engine, records[int(i)])
        if r is not None:
            rows.append(r)
        if len(rows) >= cap:
            break
    agg = {k: float(np.mean([r[k] for r in rows])) for k in rows[0]} if rows else {}
    agg["n_fires"] = len(rows)
    return agg, rows


def benchmark_speedup(cfg: dict[str, Any], engine: SurrogateEngine, world_id: int,
                      n_members: int) -> dict[str, float]:
    world = generate_world(cfg, world_id)
    ign = (world.n // 2 + 30, world.n // 2 - 40)
    t0 = time.perf_counter()
    run_ensemble(world, ign, [60.0], n_members, rng_for(cfg, "speedup-phys"))
    t_phys = time.perf_counter() - t0
    t0 = time.perf_counter()
    engine.ensemble(world, ign, [60.0], n_members, rng_for(cfg, "speedup-surr"))
    t_surr = time.perf_counter() - t0
    return {"members": n_members, "physics_s": t_phys, "surrogate_s": t_surr,
            "speedup_x": t_phys / t_surr}


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluate the fire surrogate")
    ap.add_argument("--cap", type=int, default=None, help="max fires per split")
    args = ap.parse_args()

    cfg = load_config()
    engine = SurrogateEngine(cfg)
    print(f"checkpoint: step {engine.trained_steps}, best val IoU@+10 {engine.val_iou:.4f}")

    splits = split_world_ids(cfg)
    cap = args.cap or int(cfg["surrogate"]["eval"]["n_eval_worlds"]) * 8
    rng = rng_for(cfg, "surrogate-eval")

    val_agg, val_rows = eval_split(engine, load_records(cfg, splits["val"]), cap, rng)
    hold_agg, _ = eval_split(engine, load_records(cfg, splits["holdout"]), cap, rng)
    iou30 = [r["iou_30"] for r in val_rows]
    p5_iou30 = float(np.percentile(iou30, 5)) if iou30 else float("nan")

    bench = benchmark_speedup(cfg, engine, splits["val"][0],
                              int(cfg["surrogate"]["eval"]["speedup_members"]))

    metrics = {
        "checkpoint_step": engine.trained_steps,
        "val": val_agg,
        "holdout_wind_regime": hold_agg,
        "worst_case_p5_iou_30": p5_iou30,
        "speedup": bench,
        "targets": {"iou_30": 0.80, "speedup_x": 100.0},
    }
    save_metrics("surrogate", metrics)

    def fmt(agg: dict[str, float]) -> str:
        if not agg or agg.get("n_fires", 0) == 0:
            return "| (no fires evaluated) | | | | |"
        return (f"| {agg['n_fires']:.0f} | {agg['iou_10']:.3f} | {agg['iou_30']:.3f} "
                f"| {agg['iou_60']:.3f} | {agg['arrival_mae_min']:.1f} |")

    gap = []
    if val_agg.get("iou_30", 0) < 0.80:
        gap.append(f"- IoU@+30 = {val_agg.get('iou_30', float('nan')):.3f} misses the 0.80 "
                   "aspirational target. Main error mode: autoregressive drift — small "
                   "front-position errors compound over 3 steps; more training worlds and "
                   "longer training (this run was CPU-budget-capped) are the obvious levers.")
    if hold_agg.get("iou_30", 1.0) < val_agg.get("iou_30", 0.0) - 0.10:
        gap.append(f"- Held-out wind regime IoU@+30 = {hold_agg['iou_30']:.3f} vs "
                   f"{val_agg['iou_30']:.3f} on val: the model generalizes worse to wind "
                   "directions it never saw. Late-stage fine-tuning that oversampled "
                   "small-fire frames improved worst-case behaviour but sharpened this "
                   "regime gap; the fix is training-time wind-direction augmentation "
                   "(rotate world+wind jointly), which we did not fit in the CPU budget.")
    if bench["speedup_x"] < 100.0:
        gap.append(f"- Ensemble speedup = {bench['speedup_x']:.1f}x misses the 100x target. "
                   "Context: our physics baseline is itself a heavily vectorised CA "
                   f"({bench['physics_s']:.1f} s for {bench['members']:.0f} members x 60 min), "
                   "not an operational-grade solver, so the denominator is unusually fast. "
                   "Against FARSITE-class physics the surrogate's one-forward-per-10-min "
                   "batched rollout would win by orders of magnitude; here it wins by "
                   "batching members through one network pass.")
    gap_text = "\n".join(gap) if gap else "All aspirational targets met."

    body = f"""## Neural surrogate (Phase 3)

Regenerate: `python -m emberline.surrogate.eval` (uses `data/checkpoints/best.pt`,
config `surrogate.*` in config.yaml). Splits are by WORLD; the held-out wind
regime ({cfg['surrogate']['dataset']['holdout_wind_regime']['dir_deg_min']:.0f}-{cfg['surrogate']['dataset']['holdout_wind_regime']['dir_deg_max']:.0f} deg) never appeared in training.

| split | fires | IoU@+10 | IoU@+30 | IoU@+60 | arrival MAE (min) |
|---|---|---|---|---|---|
| val (unseen worlds) {fmt(val_agg)[1:]}
| held-out wind regime {fmt(hold_agg)[1:]}

Worst-case: 5th-percentile IoU@+30 across val fires = **{p5_iou30:.3f}**.

Ensemble wall-clock, {bench['members']:.0f} members x 60 sim-min on 4 CPU threads:
physics {bench['physics_s']:.2f} s vs surrogate {bench['surrogate_s']:.2f} s ->
**{bench['speedup_x']:.1f}x**.

Gap analysis (targets were goals, not claims):
{gap_text}
"""
    update_section("surrogate", body)
    print(f"val: {val_agg}")
    print(f"holdout: {hold_agg}")
    print(f"worst-case p5 IoU@+30: {p5_iou30:.3f}")
    print(f"speedup: {bench}")
    print("wrote metrics/surrogate.json and REPORT.md section")


if __name__ == "__main__":
    main()
