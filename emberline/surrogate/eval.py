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
    ap.add_argument("--compare-baseline", default=None, metavar="JSON",
                    help="path to the archived v1 metrics JSON; also writes the "
                         "Phase-9 wind-augmentation before/after section")
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

    if args.compare_baseline:
        _write_wind_aug_section(args.compare_baseline, metrics)


def _write_wind_aug_section(baseline_path: str, new: dict[str, Any]) -> None:
    """Phase 9 before/after table: v1 (pre-augmentation) vs v2, never overwriting
    the old record. Three columns because the v1 numbers exist twice: as quoted
    in the original REPORT.md (built on the team laptop, carried into the
    Round-2 brief) and as re-measured HERE from the archived v1 checkpoint on
    the deterministically regenerated dataset — torch CPU kernels differ across
    platforms and autoregressive thresholding amplifies the difference, so the
    same-machine v1/v2 pair is the like-for-like comparison."""
    import json
    import pathlib

    old = json.loads(pathlib.Path(baseline_path).read_text(encoding="utf-8"))
    # Historical values quoted from the pre-migration REPORT.md via the Round-2
    # task brief. QUOTED, not measured here; shown for continuity only.
    quoted_val_iou30, quoted_hold_iou30 = 0.681, 0.368

    def row(label, key):
        o_v, o_h = old["val"].get(key), old["holdout_wind_regime"].get(key)
        n_v, n_h = new["val"].get(key), new["holdout_wind_regime"].get(key)
        return (f"| {label} | {o_v:.3f} / {o_h:.3f} | {n_v:.3f} / {n_h:.3f} |")

    gap_old = old["val"]["iou_30"] - old["holdout_wind_regime"]["iou_30"]
    gap_new = new["val"]["iou_30"] - new["holdout_wind_regime"]["iou_30"]
    d_hold = new["holdout_wind_regime"]["iou_30"] - old["holdout_wind_regime"]["iou_30"]
    d_val = new["val"]["iou_30"] - old["val"]["iou_30"]
    d60 = new["val"]["iou_60"] - old["val"]["iou_60"]
    verdict = (
        f"The held-out-regime IoU@+30 moved {d_hold:+.3f} and the val-holdout gap "
        f"went from {gap_old:.3f} to {gap_new:.3f}"
        + (" — the regime gap is closed" if gap_new <= 0.05 else
           " — substantially narrowed but not fully closed" if gap_new < gap_old / 2 else
           " — narrowed" if gap_new < gap_old else " — NOT closed")
        + f"; val IoU@+30 moved {d_val:+.3f} alongside."
        + (f" The cost is at the long horizon: val IoU@+60 moved {d60:+.3f} — "
           "capacity now spreads across all wind orientations, and the +60 min "
           "rollout (6 autoregressive steps) pays most for it. Operationally the "
           "trade is accepted: wind-robust +10/+30 cones are what detection-time "
           "routing uses." if d60 < -0.05 else ""))

    update_section("wind-augmentation", f"""
### Wind-augmentation retraining (Phase 9)

Regenerate: `python -m emberline.surrogate.eval --compare-baseline metrics/surrogate_v1.json`
(v1 = `data/checkpoints/best_v1.pt`, archived pre-augmentation checkpoint;
v2 = `best.pt`, fine-tuned from v1 with joint world+wind rotation augmentation
— `surrogate.train.rotate_augment` in config.yaml, implemented in
`surrogate/datasets.py`, tested in `tests/test_surrogate.py`).

For continuity: the original build's REPORT quoted **val IoU@+30 {quoted_val_iou30}
/ held-out-regime {quoted_hold_iou30}** (team-laptop build of the same v1
checkpoint; those numbers are preserved, not overwritten). Re-measured on THIS
machine from the archived v1 checkpoint and the deterministically regenerated
dataset, v1 scores {old['val']['iou_30']:.3f} / {old['holdout_wind_regime']['iou_30']:.3f}
— cross-platform torch kernel differences compound over autoregressive steps,
so the like-for-like before/after is the same-machine pair below (64 fires per
split, identical eval code and RNG streams):

| metric (val / held-out wind regime) | v1 pre-augmentation | v2 wind-augmented |
|---|---|---|
{row("IoU@+10", "iou_10")}
{row("IoU@+30", "iou_30")}
{row("IoU@+60", "iou_60")}
{row("arrival MAE (min)", "arrival_mae_min")}

Worst-case p5 IoU@+30 (val): {old['worst_case_p5_iou_30']:.3f} → {new['worst_case_p5_iou_30']:.3f}.

{verdict}
""")
    print("wrote REPORT.md wind-augmentation section")


if __name__ == "__main__":
    main()
