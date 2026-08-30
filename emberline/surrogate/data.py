"""Surrogate training data: streamed fire simulations -> per-world shards.

Sample definition
-----------------
One training sample is a pair of fire states 10 sim-minutes apart on a known
world:

* input planes (built on the fly by the Dataset, stored compactly on disk):
  ``touched_t`` (burned|burning), ``burning_t``, normalized elevation,
  fuel one-hot (5), normalized wind (u, v) at t  -> 10 channels
* target planes: ``touched_{t+10}``, ``burning_{t+10}``.

Sharding & resume
-----------------
One compressed ``.npz`` per world id holds that world's elevation, fuel and
all its fires' state stacks (uint8) + wind samples, so worlds are the atomic
unit of both storage and train/val splitting (no cross-frame leakage: frames
of one fire never straddle a split). Generation skips shards that already
exist, so it can be interrupted and resumed at any time; worlds are
deterministic in (seed, world_id) so a resumed run produces byte-identical
data.

Wind-regime holdout: worlds whose base wind direction falls in the
configured band are written like any other shard but tagged; the trainer
excludes them from BOTH train and val, and eval reports them separately as
a never-seen-regime generalization test.
"""

from __future__ import annotations

import pathlib
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any

import numpy as np

from ..config import load_config, repo_root, rng_for
from ..firesim import FireSim
from ..worldgen import World, generate_world

SNAP_MIN = 10.0  # minutes between snapshots == one surrogate step


def shard_path(cfg: dict[str, Any], world_id: int) -> pathlib.Path:
    return repo_root() / cfg["surrogate"]["dataset"]["dir"] / f"world_{world_id:04d}.npz"


def _random_ignition(world: World, rng: np.random.Generator) -> tuple[int, int]:
    """Random burnable ignition, biased upwind so fires have room to run.

    Uniform placement wastes ~half the simulations: a fire ignited near its
    downwind edge exits the map after one or two snapshots and yields almost
    no training pairs. We draw several candidates and keep the most upwind
    one (largest downwind fetch), which is also where real problem fires
    relative to a town tend to start.
    """
    from ..worldgen.fuel import URBAN, WATER

    ok = (world.fuel != WATER) & (world.fuel != URBAN)
    m = 30
    ok[:m], ok[-m:], ok[:, :m], ok[:, -m:] = False, False, False, False
    rr, cc = np.where(ok)
    u, v = world.wind.uv(0.0)
    norm = float(np.hypot(u, v)) + 1e-9
    ks = rng.integers(len(rr), size=8)
    # Projection of position onto the wind direction: smaller = more upwind.
    proj = (cc[ks] * u + rr[ks] * v) / norm
    k = ks[int(np.argmin(proj))]
    return int(rr[k]), int(cc[k])


def generate_world_shard(cfg: dict[str, Any], world_id: int) -> pathlib.Path:
    """Simulate all fires for one world and write its shard (skip if present)."""
    path = shard_path(cfg, world_id)
    if path.exists():
        return path
    path.parent.mkdir(parents=True, exist_ok=True)

    ds = cfg["surrogate"]["dataset"]
    world = generate_world(cfg, world_id)
    rng = rng_for(cfg, "surrogate-data", world_id)
    n_snap = int(ds["snapshots_per_fire"])

    fires_states, fires_winds = [], []
    for _f in range(int(ds["fires_per_world"])):
        sim = FireSim(world, np.random.default_rng(rng.integers(2**63)),
                      stochastic_ros=True)
        ig_r, ig_c = _random_ignition(world, rng)
        for dr in (0, 1):  # 2x2 patch: single-cell ignitions fizzle too often
            for dc in (0, 1):
                sim.ignite(ig_r + dr, ig_c + dc)
        states = [sim.snapshot()]
        winds = [world.wind.uv(sim.t_s)]
        for _s in range(n_snap):
            sim.run(SNAP_MIN)
            states.append(sim.snapshot())
            winds.append(world.wind.uv(sim.t_s))
            touched = sim.burned_or_burning
            # Stop early if the fire died or hit the map edge (edge effects
            # would teach the model spurious boundary behaviour).
            if not (sim.state == 1).any() or touched[0].any() or touched[-1].any() \
                    or touched[:, 0].any() or touched[:, -1].any():
                break
        fires_states.append(np.stack(states).astype(np.uint8))
        fires_winds.append(np.asarray(winds, dtype=np.float32))

    arrays: dict[str, np.ndarray] = {
        "elevation": world.elevation.astype(np.float32),
        "fuel": world.fuel.astype(np.uint8),
        "wind_base_dir_deg": np.float32(world.wind_base_dir_deg),
        "n_fires": np.int32(len(fires_states)),
    }
    for i, (s, w) in enumerate(zip(fires_states, fires_winds)):
        arrays[f"fire{i}_states"] = s
        arrays[f"fire{i}_winds"] = w

    tmp = path.with_suffix(".tmp.npz")
    np.savez_compressed(tmp, **arrays)
    tmp.rename(path)  # atomic-ish: partial writes never look like done shards
    return path


def generate_dataset(cfg: dict[str, Any], workers: int = 3) -> list[pathlib.Path]:
    """Generate all missing shards in parallel. Fully resumable."""
    ds = cfg["surrogate"]["dataset"]
    ids = list(range(1, int(ds["n_worlds"]) + 1))
    todo = [i for i in ids if not shard_path(cfg, i).exists()]
    print(f"dataset: {len(ids) - len(todo)} shards exist, generating {len(todo)}")
    if todo:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(generate_world_shard, cfg, i): i for i in todo}
            for k, fut in enumerate(as_completed(futs), 1):
                fut.result()
                if k % 10 == 0 or k == len(todo):
                    print(f"  {k}/{len(todo)} shards done", flush=True)
    return [shard_path(cfg, i) for i in ids]


def is_holdout_regime(cfg: dict[str, Any], base_dir_deg: float) -> bool:
    h = cfg["surrogate"]["dataset"]["holdout_wind_regime"]
    return float(h["dir_deg_min"]) <= base_dir_deg % 360.0 <= float(h["dir_deg_max"])


def main() -> None:
    import torch

    torch.set_num_threads(1)  # generation is numpy-bound; keep cores for the pool
    cfg = load_config()
    paths = generate_dataset(cfg, workers=max(3, __import__("os").cpu_count() - 2))
    n_pairs = 0
    for p in paths:
        with np.load(p) as z:
            n_pairs += sum(z[f"fire{i}_states"].shape[0] - 1
                           for i in range(int(z["n_fires"])))
    print(f"total (t, t+10min) pairs available: {n_pairs}")


if __name__ == "__main__":
    main()
