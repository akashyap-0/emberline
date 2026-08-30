"""World assembly: terrain + fuel + town + wind for one seeded world.

``generate_world(cfg, world_id)`` is the single entry point used everywhere
(fire sim, surrogate dataset generation, sensors, demo). ``world_id=0``
reproduces the config's canonical world; other ids draw randomized wind
regimes and fresh terrain (still fully deterministic under the global seed)
for surrogate training diversity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from ..config import rng_for
from .fuel import URBAN, WATER, make_fuel
from .terrain import make_elevation, slope_components
from .town import Town, make_town
from .wind import WindModel


@dataclass
class World:
    """Everything downstream layers need about one simulated world."""

    cfg: dict[str, Any]
    world_id: int
    elevation: np.ndarray  # (n, n) metres
    fuel: np.ndarray  # (n, n) int8 codes (see worldgen.fuel)
    town: Town
    wind: WindModel
    wind_base_dir_deg: float  # this world's mean wind direction (for regime holdout)
    wind_base_speed_ms: float
    cell_m: float

    @property
    def n(self) -> int:
        return self.elevation.shape[0]

    def slope(self) -> tuple[np.ndarray, np.ndarray]:
        return slope_components(self.elevation, self.cell_m)


def generate_world(cfg: dict[str, Any], world_id: int = 0) -> World:
    """Build a complete world. Deterministic in (cfg seed, world_id)."""
    w = cfg["world"]
    n = int(w["grid_size"])
    cell_m = float(w["cell_m"])

    elev_rng = rng_for(cfg, "terrain", world_id)
    elevation = make_elevation(n, float(w["elevation"]["relief_m"]),
                               float(w["elevation"]["beta"]), elev_rng)

    fuel_rng = rng_for(cfg, "fuel", world_id)
    fuel = make_fuel(elevation, w["fuel"], float(w["fuel"]["water_level_quantile"]),
                     float(w["fuel"]["cluster_beta"]), fuel_rng)

    town_rng = rng_for(cfg, "town", world_id)
    t = w["town"]
    town = make_town(elevation, fuel == WATER, cell_m, int(t["district_cells"]),
                     int(t["street_spacing"]), int(t["n_buildings_min"]),
                     int(t["n_buildings_max"]), int(t["n_exits"]), town_rng)

    # Roads and buildings become urban fuel (low, probabilistic ignition).
    fuel[town.road_mask] = URBAN
    fuel[town.building_mask] = URBAN

    wind_rng = rng_for(cfg, "wind", world_id)
    wc = w["wind"]
    if world_id == 0:
        base_dir = float(wc["base_dir_deg"])
        base_speed = float(wc["base_speed_ms"])
    else:
        # Randomized regimes for surrogate training worlds.
        base_dir = float(wind_rng.uniform(0.0, 360.0))
        base_speed = float(wind_rng.uniform(2.0, 14.0))
    wind = WindModel(base_speed, base_dir, float(wc["gust_sigma"]),
                     float(wc["dir_sigma_deg"]), float(wc["ou_tau_s"]), wind_rng,
                     regime_shifts=wc.get("regime_shifts") or [])

    return World(cfg=cfg, world_id=world_id, elevation=elevation, fuel=fuel,
                 town=town, wind=wind, wind_base_dir_deg=base_dir,
                 wind_base_speed_ms=base_speed, cell_m=cell_m)
