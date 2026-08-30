"""STUB: OpenStreetMap road/building adapter.

Real source: OpenStreetMap via the Overpass API (highway=* ways for the
road graph, building=* footprints), e.g. through osmnx. A real
implementation would return exactly the structures
:class:`emberline.worldgen.town.Town` holds: a networkx road graph with
``length_m``/``cells`` edge attributes rasterised onto the grid, building
footprints with nearest-access nodes, and the exit nodes where major roads
leave the area of interest.
"""

from __future__ import annotations

from typing import Any


def fetch_town(lat: float, lon: float, side_m: float, cell_m: float) -> Any:
    """Return an emberline.worldgen.town.Town for the real area.

    Raises
    ------
    NotImplementedError
        Always - this is a stub; no real data is downloaded in this repo.
    """
    raise NotImplementedError(
        "OSM adapter is a stub. Implement with osmnx.graph_from_point + "
        "features_from_point(tags={'building': True}), rasterise edges onto "
        "the grid, and mark boundary-crossing major roads as exits.")
