"""STUB: LANDFIRE fuel-model adapter.

Real source: LANDFIRE (https://landfire.gov) Scott & Burgan 40 fire behavior
fuel models (FBFM40) raster, distributed as GeoTIFF via the LANDFIRE
Product Service. A real implementation would fetch the FBFM40 raster for a
bounding box and collapse the 40 classes onto Emberline's 5-code map
(water/grass/brush/timber/urban) - or better, extend
``firesim.fuel_factor`` to the full 40-class table.
"""

from __future__ import annotations

import numpy as np


def fetch_fuel(lat: float, lon: float, side_m: float, cell_m: float) -> np.ndarray:
    """Return an (n, n) int8 fuel-code grid (emberline.worldgen.fuel codes).

    Raises
    ------
    NotImplementedError
        Always - this is a stub; no real data is downloaded in this repo.
    """
    raise NotImplementedError(
        "LANDFIRE adapter is a stub. Implement via the LANDFIRE Product "
        "Service (LFPS) job API requesting FBFM40 + canopy layers, then map "
        "FBFM40 classes onto Emberline fuel codes.")
