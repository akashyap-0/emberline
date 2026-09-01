"""Surrogate-ensemble probability calibration + temperature scaling (Phase 10).

Question: when the surrogate ensemble says "30% chance this cell burns
within 30 min", does it burn ~30% of the time? We replay eval-world fires,
run the surrogate ensemble from the true initial state, and compare
predicted burn probability against the physics outcome, binned into a
reliability curve. Perfect calibration hugs the diagonal; the plot and the
expected calibration error (ECE) land in demo/out/ and REPORT.md.

Phase 10 adds **temperature scaling**: a single scalar T fitted by NLL on a
CALIBRATION set of freshly generated worlds (ids > ``n_worlds``, so they
appear in no train/val/holdout split and were never used for IoU eval),
then applied as sigmoid(logit(p)/T). T > 1 softens over-confident ensemble
frequencies. The reported before/after ECE is measured on the SAME val
fires the original ECE came from — fit and report sets never overlap.

Run: ``python -m emberline.surrogate.calibration``
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import load_config, repo_root, rng_for
from ..report import save_metrics, update_section
from ..worldgen import generate_world
from .data import generate_world_shard
from .datasets import load_records, split_world_ids
from .rollout import SurrogateEngine


def collect_probs(cfg, engine: SurrogateEngine, world_ids: list[int], n_fires: int,
                  horizon_min: float, members: int,
                  rng: np.random.Generator) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """Per-fire (predicted P(burn), physics outcome) arrays over near-fire cells."""
    probs, outcomes = [], []
    for wid in world_ids:
        if len(probs) >= n_fires:
            break
        world = generate_world(cfg, wid)
        recs = load_records(cfg, [wid])
        for rec in recs:
            k = int(horizon_min // 10)
            if len(probs) >= n_fires or rec.states.shape[0] <= k:
                continue
            rr, cc = np.where(rec.states[0] > 0)
            ignition = (int(rr.mean()), int(cc.mean()))
            res = engine.ensemble(world, ignition, [horizon_min], members, rng)
            truth = rec.states[k] > 0
            # Sample cells near the action (all-zero wilderness would swamp bins).
            band = res.prob[0] > 0.001
            band |= truth
            probs.append(res.prob[0][band])
            outcomes.append(truth[band].astype(float))
    return probs, outcomes


def reliability(p: np.ndarray, y: np.ndarray, min_count: int = 50):
    """Binned reliability curve -> (bin mean pred, bin observed freq, counts, ECE)."""
    bins = np.linspace(0, 1, 11)
    centers, freqs, weights = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() > min_count:
            centers.append(p[m].mean())
            freqs.append(y[m].mean())
            weights.append(m.sum())
    centers, freqs, weights = map(np.asarray, (centers, freqs, weights))
    ece = float(np.sum(weights * np.abs(centers - freqs)) / weights.sum())
    return centers, freqs, weights, ece


def fit_temperature(p: np.ndarray, y: np.ndarray, members: int) -> float:
    """Temperature minimising NLL over INTERIOR ensemble frequencies.

    A 20-member ensemble emits quantised frequencies {0, 1/20, ..., 1}. The
    saturated cells (exactly 0 or 1) carry no temperature signal — their
    "logit" is whatever clip we choose, not a model opinion — yet they are
    numerous enough to dominate plain NLL (measured: T=2.59, which WORSENED
    val ECE 0.060→0.229). Fitting binned ECE directly is degenerate the other
    way: collapsing every probability to the base rate zeroes fit-set ECE
    (measured: T ran to the 6.0 grid edge, val ECE 0.330). NLL restricted to
    unsaturated cells is a proper scoring rule on exactly the cells the
    temperature can move, so it has neither failure mode. T > 1 softens
    over-confident probabilities; T < 1 sharpens under-confident ones.
    """
    interior = (p > 0) & (p < 1)
    if interior.sum() < 100:  # not enough signal to fit anything
        return 1.0
    eps = 1.0 / (2.0 * members)
    pi, yi = np.clip(p[interior], eps, 1 - eps), y[interior]
    z = np.log(pi) - np.log(1 - pi)

    def nll(t: float) -> float:
        q = np.clip(1.0 / (1.0 + np.exp(-z / t)), 1e-6, 1 - 1e-6)
        return float(-(yi * np.log(q) + (1 - yi) * np.log(1 - q)).mean())

    grid = np.geomspace(0.25, 6.0, 240)
    return float(grid[int(np.argmin([nll(t) for t in grid]))])


def apply_temperature(p: np.ndarray, temperature: float, members: int) -> np.ndarray:
    """Temperature-scale ensemble frequencies (see fit_temperature).

    Saturated frequencies (exactly 0 or 1) pass through unchanged, matching
    the fit: unanimity is a quantisation floor, and softening must not
    manufacture burn probability where no member burned.
    """
    eps = 1.0 / (2.0 * members)
    pc = np.clip(p, eps, 1 - eps)
    z = np.log(pc) - np.log(1 - pc)
    out = 1.0 / (1.0 + np.exp(-z / temperature))
    return np.where((p <= 0) | (p >= 1), p, out)


def main(n_fires: int = 12, horizon_min: float = 30.0, members: int = 20,
         n_cal_worlds: int = 8, ckpt: str | None = None,
         metrics_name: str = "calibration", write_report: bool = True) -> None:
    import pathlib

    cfg = load_config()
    engine = SurrogateEngine(cfg, pathlib.Path(ckpt) if ckpt else None)
    splits = split_world_ids(cfg)

    # --- calibration worlds: FRESH ids beyond n_worlds — in no split, never
    # used for IoU eval. Generated on demand (deterministic in seed+world_id).
    # Fit worlds and check worlds are DISJOINT (the repo's split-by-world
    # hygiene): the fit proposes a temperature, the check worlds must confirm
    # it transfers, else the identity (T=1, no scaling) ships.
    n_worlds = int(cfg["surrogate"]["dataset"]["n_worlds"])
    fit_ids = list(range(n_worlds + 1, n_worlds + 1 + n_cal_worlds))
    chk_ids = list(range(n_worlds + 1 + n_cal_worlds,
                         n_worlds + 1 + n_cal_worlds + max(4, n_cal_worlds // 2)))
    cal_ids = fit_ids + chk_ids
    for wid in cal_ids:
        generate_world_shard(cfg, wid)
    p_l, y_l = collect_probs(cfg, engine, fit_ids, n_fires, horizon_min,
                             members, rng_for(cfg, "calibration-fit"))
    used_fit = len(p_l)
    p_fit, y_fit = np.concatenate(p_l), np.concatenate(y_l)
    candidate = fit_temperature(p_fit, y_fit, members)
    pc_l, yc_l = collect_probs(cfg, engine, chk_ids, max(6, n_fires // 2),
                               horizon_min, members,
                               rng_for(cfg, "calibration-check"))
    p_chk, y_chk = np.concatenate(pc_l), np.concatenate(yc_l)
    ece_chk_raw = reliability(p_chk, y_chk)[3]
    ece_chk_cand = reliability(apply_temperature(p_chk, candidate, members), y_chk)[3]
    accepted = ece_chk_cand < ece_chk_raw - 0.005

    # --- report set: same val fires + rng stream the original ECE used.
    p_v, y_v = collect_probs(cfg, engine, splits["val"], n_fires, horizon_min,
                             members, rng_for(cfg, "calibration"))
    used = len(p_v)
    p, y = np.concatenate(p_v), np.concatenate(y_v)
    centers, freqs, weights, ece = reliability(p, y)
    # Candidate applied to val regardless of the check verdict — the report
    # and the plot always show what T WOULD do to the reported population.
    centers_c, freqs_c, weights_c, ece_val_cand = reliability(
        apply_temperature(p, candidate, members), y)

    out = repo_root() / "demo" / "out" / ("calibration_v2.png" if write_report
                                          else f"{metrics_name}.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5.5, 5.5), dpi=110)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(centers, freqs, "o-", color="#d62828",
            label=f"raw ensemble — ships (ECE {ece:.3f})")
    verdict = ("passed disjoint-world check" if accepted
               else "REJECTED by disjoint-world check")
    ax.plot(centers_c, freqs_c, "s--", color="#2a9d8f" if accepted else "#adb5bd",
            label=f"with fitted T={candidate:.2f} ({verdict})\n"
                  f"→ val ECE {ece_val_cand:.3f}")
    ax.set_xlabel(f"predicted P(burn by +{horizon_min:.0f} min)")
    ax.set_ylabel("observed burn frequency (physics)")
    ax.set_title(f"Ensemble calibration — {used} val fires")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(out)

    save_metrics(metrics_name, {
        "checkpoint": str(ckpt or "data/checkpoints/best.pt"),
        "ece": ece, "ece_val_candidate": ece_val_cand,
        "candidate_temperature": candidate, "candidate_accepted": bool(accepted),
        "shipped_temperature": None,  # foresight.temperature stays null (raw cones)
        "ece_check_raw": ece_chk_raw, "ece_check_candidate": ece_chk_cand,
        "fires": used, "fit_fires": used_fit, "check_fires": len(pc_l),
        "fit_world_ids": fit_ids, "check_world_ids": chk_ids,
        "members": members, "horizon_min": horizon_min,
        "bin_pred": centers.tolist(), "bin_freq": freqs.tolist(),
        "bin_pred_candidate": centers_c.tolist(),
        "bin_freq_candidate": freqs_c.tolist(),
        "bin_weight": weights.tolist(), "bin_weight_candidate": weights_c.tolist(),
    })
    direction = ("systematically **over-confident** (predicted probabilities exceed "
                 "observed frequencies)"
                 if float(np.mean(centers - freqs)) > 0.02 else
                 "close to the diagonal" if abs(float(np.mean(centers - freqs))) <= 0.02
                 else "under-confident (observed frequencies exceed predictions)")
    if not write_report:
        print(f"[{metrics_name}] candidate T={candidate:.3f} "
              f"({'accepted' if accepted else 'rejected'} on check worlds); "
              f"val ECE raw {ece:.3f}, with candidate {ece_val_cand:.3f} "
              f"({used} fires); wrote {out}")
        return
    from ..report import load_metrics

    v1 = load_metrics("calibration_v1")
    v1_note = (f", and re-measured HERE from the archived `best_v1.pt` it scores "
               f"**raw {v1['ece']:.3f}** (`metrics/calibration_v1.json`; same "
               "platform-variance story as the IoU numbers)" if v1 else "")
    update_section("calibration", f"""
### Surrogate ensemble calibration + temperature scaling (Phase 10)

Regenerate: `python -m emberline.surrogate.calibration`. Reliability of the
{members}-member ensemble's P(burn by +{horizon_min:.0f} min) against physics outcomes on
{used} val-world fires (cells near the fire; empty wilderness excluded). The raw
curve is {direction}.

Phase 10's temperature scaling is fit-then-verify: a scalar T is fitted on
{used_fit} fires from freshly generated fit worlds {fit_ids[0]}-{fit_ids[-1]}
(outside every train/val/holdout split, never used for IoU eval), then must
IMPROVE ECE on {len(pc_l)} fires from DISJOINT check worlds
{chk_ids[0]}-{chk_ids[-1]} or the identity (T=1, no scaling) ships instead.
sigmoid(logit(p)/T) applies to unsaturated ensemble frequencies only. The fit
objective is NLL over interior frequencies — two
naive objectives failed measurably first: plain NLL is dominated by saturated
(exactly-0/1) quantised frequencies (its T=2.59 worsened val ECE 0.060→0.229),
and direct binned-ECE minimisation is degenerate (collapsing toward the base
rate zeroes fit-set ECE; T ran to the grid edge, val ECE 0.330). Both dead
ends are kept in `fit_temperature`'s docstring.

This run: candidate T = {candidate:.2f}; disjoint-check-world ECE
{ece_chk_raw:.3f} (raw) vs {ece_chk_cand:.3f} (candidate) →
**{"check PASSED" if accepted else "check FAILED — no scaling"}**.

| | ECE on the val fires (this checkpoint, this machine) |
|---|---|
| raw ensemble — **what the system ships** | **{ece:.3f}** |
| with the fitted T = {candidate:.2f} (not shipped) | {ece_val_cand:.3f} |

{("**Transfer failure, documented rather than papered over**: on freshly "
  f"generated worlds the ensemble measures badly miscalibrated (check-world raw "
  f"ECE {ece_chk_raw:.3f}) and the fitted softening helps there, but the val "
  f"population is already near-calibrated (raw {ece:.3f}) and the same T makes "
  f"it WORSE ({ece_val_cand:.3f}). Two caveats bound this finding: each "
  "population estimate rests on only 6-12 fires (per-fire calibration variance "
  "is large), and the val split also selected the training checkpoint. "
  "Consequence: `foresight.temperature` stays null — raw cone probabilities "
  "ship." if accepted and ece_val_cand > ece else
  "The candidate did not survive its disjoint-world check, so no scaling "
  "ships; raw probabilities were already the better cone." if not accepted else
  "The fitted T transfers: consider setting `foresight.temperature` to it.")}

Baselines, for the before/after: the pre-Phase-9 checkpoint's REPORT quoted
**ECE 0.113** (raw) on its build machine{v1_note}. **Most of the calibration
fix came from the Phase-9 retraining itself** — the shipped-configuration
before/after is 0.113 (v1, quoted) → **{ece:.3f}** (v2 raw, this machine).
Curves: `demo/out/calibration_v2.png`. Temperature scaling is monotone, so
cone THRESHOLD semantics change but cone SHAPES at matched percentiles do
not; the fit/check machinery stays in `surrogate/calibration.py` for the day
real-sensor hindcasts give it a population worth fitting to.
""")
    print(f"candidate T={candidate:.3f} ({'accepted' if accepted else 'rejected'} "
          f"on check worlds); val ECE raw {ece:.3f}, with candidate {ece_val_cand:.3f} "
          f"({used} fires)")
    print(f"wrote {out} and REPORT.md section")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="Ensemble calibration + temperature fit")
    ap.add_argument("--ckpt", default=None, help="checkpoint path (default best.pt)")
    ap.add_argument("--metrics-name", default="calibration",
                    help="metrics/<name>.json to write")
    ap.add_argument("--no-report", action="store_true",
                    help="measure + save metrics only; leave REPORT.md untouched")
    a = ap.parse_args()
    main(ckpt=a.ckpt, metrics_name=a.metrics_name, write_report=not a.no_report)
