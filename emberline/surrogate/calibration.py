"""Stretch (b): surrogate-ensemble probability calibration.

Question: when the surrogate ensemble says "30% chance this cell burns
within 30 min", does it burn ~30% of the time? We replay eval-world fires,
run the surrogate ensemble from the true initial state, and compare
predicted burn probability against the physics outcome, binned into a
reliability curve. Perfect calibration hugs the diagonal; the plot and the
expected calibration error (ECE) land in demo/out/ and REPORT.md.

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
from .datasets import load_records, split_world_ids
from .rollout import SurrogateEngine


def main(n_fires: int = 12, horizon_min: float = 30.0, members: int = 20) -> None:
    cfg = load_config()
    engine = SurrogateEngine(cfg)
    splits = split_world_ids(cfg)
    rng = rng_for(cfg, "calibration")

    probs, outcomes = [], []
    used = 0
    for wid in splits["val"]:
        if used >= n_fires:
            break
        world = generate_world(cfg, wid)
        recs = load_records(cfg, [wid])
        for rec in recs:
            k = int(horizon_min // 10)
            if used >= n_fires or rec.states.shape[0] <= k:
                continue
            rr, cc = np.where(rec.states[0] > 0)
            ignition = (int(rr.mean()), int(cc.mean()))
            res = engine.ensemble(world, ignition, [horizon_min], members, rng)
            truth = rec.states[k] > 0
            # Sample cells near the action (all-zero wilderness would swamp bins).
            band = res.prob[0] > 0.001
            band |= truth
            probs.append(res.prob[0][band])
            outcomes.append(truth[band])
            used += 1

    p = np.concatenate(probs)
    y = np.concatenate(outcomes).astype(float)
    bins = np.linspace(0, 1, 11)
    centers, freqs, weights = [], [], []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & (p < hi if hi < 1 else p <= hi)
        if m.sum() > 50:
            centers.append(p[m].mean())
            freqs.append(y[m].mean())
            weights.append(m.sum())
    centers, freqs, weights = map(np.asarray, (centers, freqs, weights))
    ece = float(np.sum(weights * np.abs(centers - freqs)) / weights.sum())

    out = repo_root() / "demo" / "out" / "calibration.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5), dpi=110)
    ax.plot([0, 1], [0, 1], "--", color="gray", label="perfect")
    ax.plot(centers, freqs, "o-", color="#d62828", label="surrogate ensemble")
    ax.set_xlabel(f"predicted P(burn by +{horizon_min:.0f} min)")
    ax.set_ylabel("observed burn frequency (physics)")
    ax.set_title(f"Ensemble calibration — ECE {ece:.3f} ({used} fires)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out)

    save_metrics("calibration", {"ece": ece, "fires": used, "members": members,
                                 "horizon_min": horizon_min,
                                 "bin_pred": centers.tolist(), "bin_freq": freqs.tolist()})
    direction = ("systematically **over-confident** (predicted probabilities exceed "
                 "observed frequencies), so cone thresholds should be read as "
                 "conservative-alarmist — acceptable for warning, and fixable by widening "
                 "ensemble perturbations or temperature-scaling the member outputs"
                 if float(np.mean(centers - freqs)) > 0.02 else
                 "close to the diagonal" if abs(float(np.mean(centers - freqs))) <= 0.02 else
                 "under-confident (observed frequencies exceed predictions)")
    update_section("calibration", f"""
### Surrogate ensemble calibration (stretch)

Regenerate: `python -m emberline.surrogate.calibration`. Reliability of the
{members}-member ensemble's P(burn by +{horizon_min:.0f} min) against physics outcomes on
{used} val-world fires (cells near the fire; empty wilderness excluded):
**expected calibration error {ece:.3f}** (0 = perfect). The curve
(`demo/out/calibration.png`) is {direction}.
""")
    print(f"ECE {ece:.3f} over {used} fires; wrote {out} and REPORT.md section")


if __name__ == "__main__":
    main()
