"""Monte Carlo fire ensembles -> per-horizon burn-probability maps.

Each member perturbs (a) ignition location (Gaussian jitter, modelling
detection-bearing uncertainty), (b) mean wind direction and (c) wind speed
(modelling forecast error), and enables per-cell lognormal ROS roughness
(unresolved fuel heterogeneity). The fraction of members in which a cell has
ignited by horizon t is our estimate of P(burned by t) - exactly the
"probability cone" the Foresight layer draws and routes around.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..worldgen import World
from . import FirePerturbation, FireSim


@dataclass
class EnsembleResult:
    """Burn-probability maps: ``prob[h]`` is P(ignited by horizons_min[h])."""

    horizons_min: list[float]
    prob: np.ndarray  # (n_horizons, n, n) in [0, 1]
    n_members: int


def run_ensemble(
    world: World,
    ignition: tuple[int, int],
    horizons_min: list[float],
    n_members: int,
    base_rng: np.random.Generator,
    ens_cfg: dict[str, Any] | None = None,
    sim_factory=FireSim,
) -> EnsembleResult:
    """Run ``n_members`` perturbed sims; return burn probability per horizon.

    ``sim_factory`` lets the Foresight layer swap in the neural surrogate
    while reusing identical perturbation logic (apples-to-apples speedup
    benchmarks).
    """
    e = ens_cfg or world.cfg["firesim"]["ensemble"]
    horizons = sorted(float(h) for h in horizons_min)
    n = world.n
    counts = np.zeros((len(horizons), n, n), dtype=np.float64)

    for _ in range(n_members):
        pert = FirePerturbation(
            wind_dir_delta_deg=float(base_rng.normal(0, e["wind_dir_jitter_deg"])),
            wind_speed_scale=float(np.clip(base_rng.normal(1.0, e["wind_speed_jitter_frac"]), 0.3, 2.0)),
            ignition_dr=float(base_rng.normal(0, e["ignition_jitter_cells"])),
            ignition_dc=float(base_rng.normal(0, e["ignition_jitter_cells"])),
        )
        member_rng = np.random.default_rng(base_rng.integers(2**63))
        sim = sim_factory(world, member_rng, perturbation=pert, stochastic_ros=True)
        sim.ignite(int(round(ignition[0] + pert.ignition_dr)),
                   int(round(ignition[1] + pert.ignition_dc)))
        t = 0.0
        for h, horizon in enumerate(horizons):
            sim.run(horizon - t)
            t = horizon
            counts[h] += sim.burned_or_burning

    return EnsembleResult(horizons_min=horizons, prob=counts / n_members,
                          n_members=n_members)
