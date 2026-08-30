"""Detection dataset: labelled 60-s windows, split-by-event bookkeeping.

Fire windows come from real physics-sim fires pushed through the Gaussian
plume to nodes placed around the town; confounder and ambient windows come
from the signature library. Every window carries an ``event_id`` so the
trainer can split by event (windows of one event are near-duplicates -
splitting them across train/val would leak).

Output: one npz at ``data/detect/windows.npz`` with
X (N, 4, 60) float32 raw sensor units, y (N,) {0,1}, kind (N,) str,
event_id (N,) int. Regenerated only if missing (resume-friendly).
"""

from __future__ import annotations

import pathlib
from typing import Any

import numpy as np

from ..config import load_config, rng_for
from ..firesim import FireSim
from ..worldgen import generate_world
from . import windows_path
from ..sensors import place_nodes, sample_series
from ..sensors.events import CONFOUNDERS, fire_added_series, make_confounder
from ..sensors.plume import plume_concentration

WINDOW_S = 60
FIRE_PM_GATE = 3.0  # ug/m3 mean added PM for a window to count as 'fire smoke present'


def simulate_fire_event(cfg: dict[str, Any], world, nodes, rng: np.random.Generator,
                        minutes: float = 40.0) -> np.ndarray:
    """Plume PM2.5 series (n_nodes, T) at 1 Hz from one simulated fire.

    Ignition is placed upwind of a random node at 300-900 m so its plume
    advects across the network, as a threatening real ignition would.
    """
    pl = cfg["sensors"]["plume"]
    node = nodes[int(rng.integers(len(nodes)))]
    u, v = world.wind.uv(0.0)
    norm = np.hypot(u, v) + 1e-9
    dist = rng.uniform(30, 90)  # cells
    ig_r = int(np.clip(node.r - dist * v / norm + rng.normal(0, 8), 2, world.n - 3))
    ig_c = int(np.clip(node.c - dist * u / norm + rng.normal(0, 8), 2, world.n - 3))

    sim = FireSim(world, np.random.default_rng(rng.integers(2**63)), stochastic_ros=True)
    for dr in (0, 1):
        for dc in (0, 1):
            sim.ignite(ig_r + dr, ig_c + dc)

    sensor_rc = np.array([[n.r, n.c] for n in nodes])
    n_steps = int(minutes * 60 / sim.dt_s)
    conc_steps = np.zeros((len(nodes), n_steps + 1))
    for k in range(n_steps):
        burning = np.argwhere(sim.state == 1)
        if len(burning) > 400:  # plume cost control: subsample sources
            keep = rng.choice(len(burning), 400, replace=False)
            scale = len(burning) / 400
            burning = burning[keep]
        else:
            scale = 1.0
        conc_steps[:, k] = scale * plume_concentration(
            burning, sensor_rc, world.wind.uv(sim.t_s), world.cell_m,
            float(pl["q_fire"]), float(pl["sigma_y_coef"]), float(pl["sigma_z_coef"]))
        sim.step()
    conc_steps[:, -1] = conc_steps[:, -2]

    t_step = np.arange(n_steps + 1) * sim.dt_s
    t_1hz = np.arange(0, minutes * 60)
    return np.stack([np.interp(t_1hz, t_step, c) for c in conc_steps])


def build_windows(cfg: dict[str, Any]) -> pathlib.Path:
    out = windows_path(cfg)
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)

    d = cfg["detect"]["dataset"]
    noise = cfg["sensors"]["noise"]
    rng = rng_for(cfg, "detect-data")
    X, y, kinds, event_ids = [], [], [], []
    eid = 0

    # Worlds/nodes reused across fire events (10 distinct layouts).
    layouts = []
    for i in range(10):
        w = generate_world(cfg, 300 + i)
        layouts.append((w, place_nodes(w, int(cfg["sensors"]["n_nodes"]),
                                       rng_for(cfg, "detect-nodes", i))))

    def harvest(added: np.ndarray, node, kind: str, label: int, event_id: int,
                t0_day: float, cap: int) -> None:
        """Cut 60-s windows out of one node's measured series."""
        T = added.shape[1]
        t_abs = t0_day + np.arange(T)
        series = sample_series(node, t_abs, added, noise, rng)
        starts = np.arange(0, T - WINDOW_S, WINDOW_S)
        if label == 1:  # keep only windows where smoke is actually present
            ok = [s for s in starts if added[0, s : s + WINDOW_S].mean() >= FIRE_PM_GATE]
        else:
            ok = list(starts)
        rng.shuffle(ok)
        for s in ok[:cap]:
            X.append(series[:, s : s + WINDOW_S].astype(np.float32))
            y.append(label)
            kinds.append(kind)
            event_ids.append(event_id)

    # --- fire events ------------------------------------------------------
    for i in range(int(d["n_fire_events"])):
        world, nodes = layouts[i % len(layouts)]
        conc = simulate_fire_event(cfg, world, nodes, rng)
        t0_day = float(rng.uniform(0, 86400 - conc.shape[1]))
        hit_nodes = np.argsort(conc.mean(axis=1))[::-1][:3]
        for ni in hit_nodes:
            if conc[ni].max() < FIRE_PM_GATE:
                continue
            harvest(fire_added_series(conc[ni], rng), nodes[ni], "fire", 1, eid,
                    t0_day, cap=8)
        eid += 1
        if (i + 1) % 40 == 0:
            print(f"  fire events {i + 1}/{d['n_fire_events']}", flush=True)

    # --- confounder events ------------------------------------------------
    n_conf = int(d["n_confounder_events"])
    for i in range(n_conf):
        kind = CONFOUNDERS[i % len(CONFOUNDERS)]
        world, nodes = layouts[i % len(layouts)]
        spec = make_confounder(kind, len(nodes), rng)
        T = int(min(spec.duration_s, 3600))
        t_rel = np.arange(T).astype(float)
        # Evening for stoves, morning for fog, anytime otherwise.
        t0_day = {"wood_stove": rng.uniform(17, 22) * 3600,
                  "fog": rng.uniform(3, 7) * 3600}.get(kind, rng.uniform(0, 20) * 3600)
        for ni in spec.scope[:3]:
            harvest(spec.envelope(ni, t_rel), nodes[ni], kind, 0, eid, t0_day, cap=8)
        eid += 1

    # --- ambient windows (their own event ids; used for balance) -----------
    world, nodes = layouts[0]
    for i in range(int(d["n_ambient_windows"]) // 4):
        node = nodes[int(rng.integers(len(nodes)))]
        t0_day = float(rng.uniform(0, 86400 - 5 * WINDOW_S))
        added = np.zeros((4, 4 * WINDOW_S + WINDOW_S))
        harvest(added, node, "ambient", 0, eid, t0_day, cap=4)
        eid += 1

    Xa = np.stack(X)
    np.savez_compressed(out, X=Xa, y=np.array(y, dtype=np.int64),
                        kind=np.array(kinds), event_id=np.array(event_ids, dtype=np.int64))
    print(f"detect dataset: {len(Xa)} windows "
          f"({int(sum(y))} fire / {len(y) - int(sum(y))} non-fire) -> {out}")
    return out


def main() -> None:
    build_windows(load_config())


if __name__ == "__main__":
    main()
