"""Fuel-type map generation.

Fuel classes (int8 codes, also used one-hot by the surrogate):

====  ======  =============================================================
code  name    rationale
====  ======  =============================================================
0     water   lowest-elevation basin cells (flood-fill by quantile); blocks
              fire entirely.
1     grass   fast, light surface fuel (Rothermel fuel model 1-ish).
2     brush   moderate shrub fuel (model 5/6-ish).
3     timber  slower surface spread under canopy (model 8/9-ish). We model
              SURFACE spread only - no crown fire.
4     urban   building/road cells; low spread rate and probabilistic
              ignition (structures ignite from ember/radiant exposure, not
              modelled explicitly).
====  ======  =============================================================

Vegetation classes are assigned by thresholding a second, independent
fractal noise field so that classes form realistic contiguous patches
(the field's spectral slope controls patch size), with thresholds chosen by
quantile to hit the configured area fractions. Elevation biases the noise
slightly (timber favours higher, wetter ground) which mimics real
vegetation banding without extra machinery.
"""

from __future__ import annotations

import numpy as np

from .terrain import fractal_field

WATER, GRASS, BRUSH, TIMBER, URBAN = 0, 1, 2, 3, 4
FUEL_NAMES = {WATER: "water", GRASS: "grass", BRUSH: "brush", TIMBER: "timber", URBAN: "urban"}
N_FUEL_CLASSES = 5


def make_fuel(
    elevation: np.ndarray,
    fractions: dict[str, float],
    water_level_quantile: float,
    cluster_beta: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Fuel map (int8) matching target area fractions for grass/brush/timber.

    Water is carved first from the lowest-elevation quantile; the remaining
    land is split among vegetation classes by quantiles of an
    elevation-biased fractal field, guaranteeing the configured fractions
    hold on land cells (up to discretisation).
    """
    n = elevation.shape[0]
    fuel = np.full((n, n), GRASS, dtype=np.int8)

    water_level = np.quantile(elevation, water_level_quantile)
    water_mask = elevation <= water_level
    fuel[water_mask] = WATER

    veg_noise = fractal_field(n, cluster_beta, rng)
    elev_norm = (elevation - elevation.mean()) / (elevation.std() + 1e-12)
    score = veg_noise + 0.6 * elev_norm  # timber banding toward high ground

    land = ~water_mask
    land_scores = score[land]
    g, b = fractions["grass"], fractions["brush"]
    total = g + b + fractions["timber"]
    q_grass = np.quantile(land_scores, g / total)
    q_brush = np.quantile(land_scores, (g + b) / total)

    fuel[land & (score <= q_grass)] = GRASS
    fuel[land & (score > q_grass) & (score <= q_brush)] = BRUSH
    fuel[land & (score > q_brush)] = TIMBER
    return fuel
