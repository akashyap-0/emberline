"""Phase 1 acceptance tests: map statistics, road connectivity, reachability."""

from __future__ import annotations

import networkx as nx
import numpy as np
import pytest

from emberline.config import load_config
from emberline.worldgen import generate_world
from emberline.worldgen.fuel import BRUSH, GRASS, TIMBER, URBAN, WATER


@pytest.fixture(scope="module")
def world():
    return generate_world(load_config(), world_id=0)


def test_determinism(world):
    w2 = generate_world(load_config(), world_id=0)
    assert np.array_equal(world.elevation, w2.elevation)
    assert np.array_equal(world.fuel, w2.fuel)
    assert len(world.town.buildings) == len(w2.town.buildings)


def test_fuel_fractions(world):
    cfg = load_config()["world"]["fuel"]
    n_cells = world.fuel.size
    water_frac = (world.fuel == WATER).mean()
    assert 0.5 * cfg["water_level_quantile"] <= water_frac <= 2.0 * cfg["water_level_quantile"]
    # Vegetation fractions on non-water, non-urban land should be near targets.
    land = np.isin(world.fuel, [GRASS, BRUSH, TIMBER])
    total_veg = land.sum()
    for code, key in [(GRASS, "grass"), (BRUSH, "brush"), (TIMBER, "timber")]:
        frac = (world.fuel == code).sum() / total_veg
        target = cfg[key] / (cfg["grass"] + cfg["brush"] + cfg["timber"])
        assert abs(frac - target) < 0.10, f"{key}: {frac:.3f} vs {target:.3f}"
    assert (world.fuel == URBAN).sum() > 0
    assert (world.fuel == URBAN).sum() < 0.2 * n_cells


def test_building_count(world):
    t = load_config()["world"]["town"]
    nb = len(world.town.buildings)
    assert t["n_buildings_min"] <= nb <= t["n_buildings_max"]


def test_road_graph_connected(world):
    g = world.town.roads
    assert nx.is_connected(g)
    n_exits = load_config()["world"]["town"]["n_exits"]
    assert len(world.town.exits) == n_exits
    n = world.n
    for er, ec in world.town.exits:
        assert er in (0, n - 1) or ec in (0, n - 1), "exit must reach the map edge"


def test_all_buildings_reachable_from_exit(world):
    g = world.town.roads
    exit_set = set(world.town.exits)
    for b in world.town.buildings:
        assert b.access in g
        assert any(nx.has_path(g, b.access, e) for e in exit_set)


def test_wind_model(world):
    speeds = [world.wind.at(t)[0] for t in np.linspace(0, 3600, 200)]
    assert all(s >= 0 for s in speeds)
    assert np.std(speeds) > 0.01, "gusts should vary wind speed"
    base = np.deg2rad(world.wind_base_dir_deg)
    dirs = np.array([world.wind.at(t)[1] for t in np.linspace(0, 3600, 200)])
    assert np.abs(np.angle(np.exp(1j * (dirs - base)))).mean() < np.deg2rad(45)
    # Runtime shift moves the mean direction.
    world2 = generate_world(load_config(), world_id=0)
    d_before = world2.wind.at(5000)[1]
    world2.wind.apply_shift(4000, dir_delta_deg=40)
    d_after = world2.wind.at(5000)[1]
    assert np.isclose(np.rad2deg(d_after - d_before), 40.0, atol=1e-6)


def test_randomized_worlds_differ():
    cfg = load_config()
    w1, w2 = generate_world(cfg, 1), generate_world(cfg, 2)
    assert not np.array_equal(w1.elevation, w2.elevation)
    assert w1.wind_base_dir_deg != w2.wind_base_dir_deg
