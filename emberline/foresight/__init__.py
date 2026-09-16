"""Foresight: digital-twin decision layer.

On a Tier-1+ mesh event: estimate the ignition point from detecting nodes'
bearings, run the surrogate ensemble (physics fallback) on the twin, draw
probability cones at +10/+30/+60 min, route evacuations around the cone,
and draft a CAP-style alert for HUMAN APPROVAL. Every routing output is
labelled ADVISORY - this layer supports a decision, it does not make one.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from scipy import ndimage

from ..config import rng_for
from ..firesim.ensemble import EnsembleResult, run_ensemble
from ..worldgen import World


def estimate_ignition(
    detections: list[tuple[tuple[int, int], float]],  # ((r, c), bearing_deg to smoke source)
    world: World,
    fallback_range_m: float = 400.0,
) -> tuple[int, int]:
    """Ignition estimate from node positions + bearings.

    With >= 2 bearings we take the least-squares intersection of the bearing
    rays (classic two-station fire triangulation); with one, we project a
    plausible plume-travel distance up the bearing. Bearings point FROM the
    node TOWARD the smoke source (i.e. upwind).
    """
    n = world.n
    if len(detections) >= 2:
        # Least squares: minimise sum of squared distances to each ray's line.
        a_mat = np.zeros((2, 2))
        b_vec = np.zeros(2)
        for (r, c), bearing in detections[:4]:
            th = np.deg2rad(bearing)
            d = np.array([np.cos(th), np.sin(th)])  # (x, y)
            p = np.array([c, r], dtype=float)
            proj = np.eye(2) - np.outer(d, d)
            a_mat += proj
            b_vec += proj @ p
        # Near-parallel bearings make the system ill-conditioned and the
        # "intersection" flies off the map - fall back to range projection.
        if np.linalg.cond(a_mat) > 1e4:
            (r, c), bearing = detections[0]
            th = np.deg2rad(bearing)
            rr = r + np.sin(th) * fallback_range_m / world.cell_m
            cc = c + np.cos(th) * fallback_range_m / world.cell_m
        else:
            xy = np.linalg.solve(a_mat, b_vec)
            cc, rr = xy
    else:
        (r, c), bearing = detections[0]
        th = np.deg2rad(bearing)
        rr = r + np.sin(th) * fallback_range_m / world.cell_m
        cc = c + np.cos(th) * fallback_range_m / world.cell_m
    return int(np.clip(rr, 1, n - 2)), int(np.clip(cc, 1, n - 2))


@dataclass
class Cone:
    """Probability cones at each horizon + the routing exclusion mask."""

    result: EnsembleResult
    thresholds: list[float]
    margin_cells: int

    def mask(self, horizon_idx: int, threshold: float | None = None) -> np.ndarray:
        """Cells with P(burn) over threshold, dilated by the safety margin."""
        thr = threshold if threshold is not None else min(self.thresholds)
        raw = self.result.prob[horizon_idx] >= thr
        if self.margin_cells > 0:
            raw = ndimage.binary_dilation(raw, iterations=self.margin_cells)
        return raw

    def union_mask(self) -> np.ndarray:
        """Warn-area mask (CAP polygon): any horizon over the lowest threshold."""
        raw = (self.result.prob >= min(self.thresholds)).any(axis=0)
        return ndimage.binary_dilation(raw, iterations=self.margin_cells)

    def routing_mask(self, horizon_min: float = 30.0, threshold: float = 0.30) -> np.ndarray:
        """Road-exclusion mask: where fire is LIKELY within the evacuation window.

        Evacuation happens over the next tens of minutes, so roads are cut
        where P(burn by +horizon) exceeds a mid threshold - cutting on the
        +60 min low-probability cone would (correctly but uselessly) declare
        most of a downwind town unroutable while people still have time to
        drive out of it.
        """
        idx = int(np.argmin([abs(h - horizon_min) for h in self.result.horizons_min]))
        raw = self.result.prob[idx] >= threshold
        return ndimage.binary_dilation(raw, iterations=self.margin_cells)


class Foresight:
    """Forecast engine wrapper: surrogate ensemble, physics fallback."""

    def __init__(self, cfg: dict[str, Any], world: World) -> None:
        self.cfg = cfg
        self.world = world
        self.engine = None
        try:
            from ..surrogate.rollout import SurrogateEngine

            if SurrogateEngine.available(cfg):
                self.engine = SurrogateEngine(cfg)
        except Exception:
            self.engine = None
        self.backend = "surrogate" if self.engine is not None else "physics"

    def forecast(self, ignition: tuple[int, int], t0_s: float = 0.0,
                 n_members: int | None = None) -> Cone:
        f = self.cfg["foresight"]
        horizons = [float(h) for h in f["horizons_min"]]
        members = int(n_members or f["ensemble_members"])
        rng = rng_for(self.cfg, "foresight-forecast", int(t0_s))
        if self.engine is not None:
            res = self.engine.ensemble(self.world, ignition, horizons, members, rng,
                                       t0_s=t0_s)
        else:
            res = run_ensemble(self.world, ignition, horizons, members, rng)
        # Optional Phase-10 temperature scaling of ensemble burn probabilities
        # (fit by emberline.surrogate.calibration; off by default so cone
        # threshold semantics only change when explicitly configured).
        temp = f.get("temperature")
        if temp:
            from ..surrogate.calibration import apply_temperature

            res = EnsembleResult(horizons_min=res.horizons_min,
                                 prob=apply_temperature(res.prob, float(temp), members),
                                 n_members=res.n_members)
        return Cone(result=res, thresholds=[float(t) for t in f["cone_thresholds"]],
                    margin_cells=int(f["cone_margin_cells"]))
