"""Synthetic sensor nodes: placement, baselines, noise, drift.

Each node samples PM2.5 (ug/m3), VOC (index), temperature (C) and relative
humidity (%) at 1 Hz. Signal = diurnal baseline + event contributions
(fire plume or confounder) + white noise + slow drift.

Baseline assumptions (documented so they can be defended or replaced):
* temperature: sinusoid peaking ~16:00, 8 C amplitude around 15 C;
* RH: anti-correlated sinusoid (moisture roughly constant, RH follows temp);
* PM2.5: ~8 ug/m3 background with morning/evening traffic bumps;
* VOC: small diurnal ripple around 0.3;
* drift: each node's PM channel drifts linearly (random sign, ~0.5 ug/m3
  per day) - cheap sensors do this, and it is exactly what the mesh layer's
  health scores are for.

Node placement: a ring just outside the town district (the asset we defend)
plus outposts pushed toward the prevailing upwind sector, where early
detection buys the most warning time.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..worldgen import World


@dataclass
class SensorNode:
    """Static description of one deployed node."""

    node_id: str
    r: int
    c: int
    drift_sign: float  # +1/-1, applied to PM drift
    battery_v: float = 3.9


def place_nodes(world: World, n_nodes: int, rng: np.random.Generator) -> list[SensorNode]:
    """Ring around town + upwind outposts. Deterministic under the seed."""
    r0, c0, r1, c1 = world.town.district
    cr, cc = (r0 + r1) / 2, (c0 + c1) / 2
    half = (r1 - r0) / 2
    nodes: list[SensorNode] = []
    n_ring = max(4, int(n_nodes * 0.7))
    for k in range(n_ring):
        ang = 2 * np.pi * k / n_ring
        rad = half + rng.uniform(15, 35)  # 150-350 m beyond the district edge
        r = int(np.clip(cr + rad * np.sin(ang), 2, world.n - 3))
        c = int(np.clip(cc + rad * np.cos(ang), 2, world.n - 3))
        nodes.append(SensorNode(f"N{k}", r, c, float(rng.choice([-1, 1]))))
    # Outposts toward the prevailing upwind direction (fires come from there).
    u, v = world.wind.uv(0.0)
    norm = np.hypot(u, v) + 1e-9
    for k in range(n_ring, n_nodes):
        dist = rng.uniform(60, 110)  # 600-1100 m out
        jitter = rng.normal(0, 12, size=2)
        r = int(np.clip(cr - dist * v / norm + jitter[0], 2, world.n - 3))
        c = int(np.clip(cc - dist * u / norm + jitter[1], 2, world.n - 3))
        nodes.append(SensorNode(f"N{k}", r, c, float(rng.choice([-1, 1]))))
    return nodes


CHANNELS = ["pm25", "voc", "temp", "rh"]


def baseline(t_s: np.ndarray) -> np.ndarray:
    """(4, len(t)) diurnal baselines; t_s = seconds since local midnight."""
    day = 86400.0
    temp = 15.0 + 8.0 * np.sin(2 * np.pi * (t_s / day - 5 / 12))  # peak at 16:00
    rh = np.clip(60.0 - 2.5 * (temp - 15.0), 15.0, 100.0)
    traffic = (np.exp(-0.5 * ((t_s / 3600 - 8) / 1.2) ** 2)
               + np.exp(-0.5 * ((t_s / 3600 - 20) / 1.5) ** 2))
    pm = 8.0 + 4.0 * traffic
    voc = 0.30 + 0.08 * traffic
    return np.stack([pm, voc, temp, rh])


def sample_series(
    node: SensorNode,
    t_s: np.ndarray,
    added: np.ndarray,  # (4, len(t)) event contribution (pm, voc, dT, dRH)
    noise_cfg: dict,
    rng: np.random.Generator,
) -> np.ndarray:
    """Full measured series for one node: baseline + event + noise + drift."""
    sig = baseline(t_s) + added
    sig[0] += node.drift_sign * float(noise_cfg["drift_per_day"]) * (t_s / 86400.0)
    sig[0] += rng.normal(0, float(noise_cfg["pm25_sigma"]), t_s.shape)
    sig[1] += rng.normal(0, float(noise_cfg["voc_sigma"]), t_s.shape)
    sig[2] += rng.normal(0, float(noise_cfg["temp_sigma"]), t_s.shape)
    sig[3] += rng.normal(0, float(noise_cfg["rh_sigma"]), t_s.shape)
    sig[0] = np.clip(sig[0], 0, None)
    sig[1] = np.clip(sig[1], 0, None)
    sig[3] = np.clip(sig[3], 0, 100)
    return sig
