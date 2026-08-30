"""STUB: USGS 3DEP elevation adapter.

Real source: USGS 3D Elevation Program (3DEP), e.g. the 1/3 arc-second
(~10 m) DEM tiles served via The National Map API
(https://apps.nationalmap.gov/tnmaccess/). A real implementation would
download the GeoTIFF tile(s) covering a bounding box, reproject to a local
metric grid (UTM), and resample to the configured cell size - returning the
same ``(n, n) float64 metres`` array that
:func:`emberline.worldgen.terrain.make_elevation` produces synthetically.
"""

from __future__ import annotations

import numpy as np


def fetch_elevation(lat: float, lon: float, side_m: float, cell_m: float) -> np.ndarray:
    """Return an (n, n) elevation grid in metres centred on (lat, lon).

    Raises
    ------
    NotImplementedError
        Always - this is a stub; no real data is downloaded in this repo.
    """
    raise NotImplementedError(
        "USGS 3DEP adapter is a stub. Implement by querying The National Map "
        "TNM Access API for 1/3 arc-second DEM tiles, reprojecting to UTM, "
        "and resampling to cell_m.")
