"""Phase 2 acceptance tests: conservation, wind/slope response, speed, water."""

from __future__ import annotations

import time

import numpy as np
import pytest

from emberline.config import load_config, rng_for
from emberline.firesim import BURNED, FireSim
from emberline.firesim.ensemble import run_ensemble
from emberline.worldgen import generate_world
from emberline.worldgen.fuel import GRASS, WATER


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def world(cfg):
    return generate_world(cfg, world_id=0)


def _flat_grass_world(cfg, base_dir_deg=0.0, speed=8.0):
    """Synthetic world: flat terrain, all grass, steady wind toward +x."""
    w = generate_world(cfg, world_id=0)
    w.elevation = np.zeros_like(w.elevation)
    w.fuel = np.full_like(w.fuel, GRASS)
    from emberline.worldgen.wind import WindModel

    w.wind = WindModel(speed, base_dir_deg, 0.0, 0.0, 300.0,
                       np.random.default_rng(0))
    return w


def test_conservation_burned_never_unburns(cfg, world):
    sim = FireSim(world, rng_for(cfg, "t-conserve"))
    sim.ignite(128, 128)
    prev_touched = sim.burned_or_burning.copy()
    prev_burned = sim.state == BURNED
    for _ in range(60):
        sim.step()
        touched = sim.burned_or_burning
        burned = sim.state == BURNED
        assert not (prev_touched & ~touched).any(), "an ignited cell reverted to unburned"
        assert not (prev_burned & (sim.state != BURNED)).any(), "a burned cell changed state"
        prev_touched, prev_burned = touched.copy(), burned.copy()


def test_wind_drives_downwind_spread(cfg):
    w = _flat_grass_world(cfg, base_dir_deg=0.0)  # wind toward +x (east)
    sim = FireSim(w, rng_for(cfg, "t-wind"))
    sim.ignite(128, 128)
    sim.run(45)
    rr, cc = np.where(sim.burned_or_burning)
    assert cc.mean() - 128 > 10, "burned centroid should be displaced downwind (east)"
    assert abs(rr.mean() - 128) < 8, "little crosswind displacement expected"
    east = (cc - 128).max()
    west = (128 - cc).max()
    assert east > 2.0 * west, f"downwind run {east} should dominate upwind {west}"


def test_upslope_spread_without_wind(cfg):
    w = _flat_grass_world(cfg, speed=0.0)
    n = w.n
    # Terrain rising to the north at 30% grade.
    w.elevation = np.arange(n, dtype=float)[:, None].repeat(n, 1) * w.cell_m * 0.30
    sim = FireSim(w, rng_for(cfg, "t-slope"))
    sim.ignite(128, 128)
    sim.run(45)
    rr, _ = np.where(sim.burned_or_burning)
    assert rr.mean() > 130, "fire should run upslope (north)"


def test_water_blocks(cfg):
    w = _flat_grass_world(cfg, base_dir_deg=0.0)
    w.fuel[:, 140:150] = WATER  # 100 m wide channel downwind of ignition
    sim = FireSim(w, rng_for(cfg, "t-water"))
    sim.ignite(128, 130)
    sim.run(60)
    assert not sim.burned_or_burning[:, 150:].any(), "fire must not cross water"
    assert not sim.burned_or_burning[:, 140:150].any(), "water cells must never ignite"


def test_speed_60min_under_5s(cfg, world):
    sim = FireSim(world, rng_for(cfg, "t-speed"))
    sim.ignite(60, 60)
    t0 = time.perf_counter()
    sim.run(60)
    assert time.perf_counter() - t0 < 5.0


def test_ensemble_probability_maps(cfg, world):
    res = run_ensemble(world, (100, 100), [10, 30], 12, rng_for(cfg, "t-ens"))
    assert res.prob.shape == (2, world.n, world.n)
    assert res.prob.min() >= 0 and res.prob.max() <= 1
    assert (res.prob[1] >= res.prob[0] - 1e-12).all(), "P(burned) monotone in horizon"
    assert res.prob[1].sum() > res.prob[0].sum()


def test_stochastic_members_differ(cfg, world):
    a = FireSim(world, np.random.default_rng(1), stochastic_ros=True)
    b = FireSim(world, np.random.default_rng(2), stochastic_ros=True)
    for s in (a, b):
        s.ignite(100, 100)
        s.run(30)
    assert (a.state != b.state).any()
