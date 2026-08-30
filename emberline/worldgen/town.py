"""Procedural town: building footprints on a road graph with exit roads.

Design
------
* **Site selection**: the town district (a square of ``district_cells``) is
  placed on the flattest, driest candidate window on the map - real towns
  sit on flat valley ground, and it keeps roads plausible.
* **Street grid**: intersections every ``street_spacing`` cells inside the
  district, connected as a lattice (networkx). Real small towns are mostly
  grid-planned; a lattice also guarantees internal connectivity by
  construction, which the tests then verify independently.
* **Exit roads**: ``n_exits`` (2-4) straight roads from the district border
  to the map edge on distinct sides. These are the evacuation bottlenecks
  the Foresight layer routes over.
* **Buildings**: 150-400 single-cell (10 m x 10 m) footprints sampled from
  lots adjacent to streets, each recorded with its nearest road-graph node
  ("access node") so evacuation routing can start from any building.

Road edges carry ``length_m`` and the raster ``cells`` they cover, so later
layers can (a) burn roads into the fuel map as urban cells and (b) test
whether a fire-probability cone cuts a road.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import networkx as nx
import numpy as np

from .terrain import slope_components

Node = tuple[int, int]  # (row, col)


@dataclass
class Building:
    """One building footprint cell and its road access node."""

    r: int
    c: int
    access: Node


@dataclass
class Town:
    """Road graph + buildings + exit nodes for one generated town."""

    roads: nx.Graph
    road_mask: np.ndarray  # bool (n, n): True where a road runs
    buildings: list[Building]
    exits: list[Node]  # graph nodes lying on the map edge
    district: tuple[int, int, int, int] = field(default=(0, 0, 0, 0))  # r0, c0, r1, c1

    @property
    def building_mask(self) -> np.ndarray:
        m = np.zeros_like(self.road_mask)
        for b in self.buildings:
            m[b.r, b.c] = True
        return m


def _line_cells(a: Node, b: Node) -> list[Node]:
    """Raster cells along an axis-aligned segment from a to b (inclusive)."""
    (r0, c0), (r1, c1) = a, b
    if r0 == r1:
        step = 1 if c1 >= c0 else -1
        return [(r0, c) for c in range(c0, c1 + step, step)]
    if c0 == c1:
        step = 1 if r1 >= r0 else -1
        return [(r, c0) for r in range(r0, r1 + step, step)]
    raise ValueError("only axis-aligned road segments are supported")


def _select_site(elevation: np.ndarray, water_mask: np.ndarray, side: int, cell_m: float,
                 rng: np.random.Generator) -> tuple[int, int]:
    """Top-left corner of the flattest, driest district window.

    Scans a coarse lattice of candidate windows and scores each by mean
    slope magnitude plus a heavy water-fraction penalty. A small random
    tie-break keeps different world seeds from always picking pixel-identical
    sites on similar terrain.
    """
    n = elevation.shape[0]
    dz_dx, dz_dy = slope_components(elevation, cell_m)
    slope = np.hypot(dz_dx, dz_dy)
    margin = 8  # keep district off the absolute edge so exits have length
    best, best_score = None, np.inf
    for r0 in range(margin, n - side - margin, 8):
        for c0 in range(margin, n - side - margin, 8):
            win = np.s_[r0 : r0 + side, c0 : c0 + side]
            score = slope[win].mean() + 10.0 * water_mask[win].mean() + rng.uniform(0, 1e-3)
            if score < best_score:
                best_score, best = score, (r0, c0)
    assert best is not None
    return best


def make_town(
    elevation: np.ndarray,
    water_mask: np.ndarray,
    cell_m: float,
    district_cells: int,
    street_spacing: int,
    n_buildings_min: int,
    n_buildings_max: int,
    n_exits: int,
    rng: np.random.Generator,
) -> Town:
    """Generate the town: site, street lattice, exit roads, buildings."""
    n = elevation.shape[0]
    r0, c0 = _select_site(elevation, water_mask, district_cells, cell_m, rng)
    r1, c1 = r0 + district_cells, c0 + district_cells

    rows = list(range(r0, r1 + 1, street_spacing))
    cols = list(range(c0, c1 + 1, street_spacing))

    g = nx.Graph()
    road_mask = np.zeros((n, n), dtype=bool)

    def add_edge(a: Node, b: Node, kind: str) -> None:
        cells = _line_cells(a, b)
        for r, c in cells:
            road_mask[r, c] = True
        length_m = (len(cells) - 1) * cell_m
        g.add_edge(a, b, length_m=length_m, cells=cells, kind=kind)

    # Street lattice.
    for i, r in enumerate(rows):
        for j, c in enumerate(cols):
            if j + 1 < len(cols):
                add_edge((r, c), (r, cols[j + 1]), "street")
            if i + 1 < len(rows):
                add_edge((r, c), (rows[i + 1], c), "street")

    # Exit roads on distinct sides. Sides: 0=N(top,row 0), 1=E, 2=S, 3=W.
    sides = rng.choice(4, size=n_exits, replace=False)
    exits: list[Node] = []
    for side in sides:
        if side == 0:  # north: extend a column road up to row 0
            col = int(rng.choice(cols))
            a, b = (r0, col), (0, col)
        elif side == 2:  # south
            col = int(rng.choice(cols))
            a, b = (rows[-1], col), (n - 1, col)
        elif side == 1:  # east
            row = int(rng.choice(rows))
            a, b = (row, cols[-1]), (row, n - 1)
        else:  # west
            row = int(rng.choice(rows))
            a, b = (row, c0), (row, 0)
        add_edge(a, b, "exit")
        exits.append(b)

    # Buildings: lots one cell off a street, inside the district, not on water.
    candidates: list[Node] = []
    for r in range(r0, rows[-1] + 1):
        for c in range(c0, cols[-1] + 1):
            if road_mask[r, c] or water_mask[r, c]:
                continue
            nb = [(r + dr, c + dc) for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1))]
            if any(0 <= rr < n and 0 <= cc < n and road_mask[rr, cc] for rr, cc in nb):
                candidates.append((r, c))
    n_buildings = int(rng.integers(n_buildings_min, n_buildings_max + 1))
    n_buildings = min(n_buildings, len(candidates))
    idx = rng.choice(len(candidates), size=n_buildings, replace=False)

    nodes = np.array(list(g.nodes))
    buildings: list[Building] = []
    for k in idx:
        r, c = candidates[k]
        d2 = (nodes[:, 0] - r) ** 2 + (nodes[:, 1] - c) ** 2
        access = tuple(int(v) for v in nodes[np.argmin(d2)])
        buildings.append(Building(r, c, access))  # type: ignore[arg-type]

    return Town(roads=g, road_mask=road_mask, buildings=buildings, exits=exits,
                district=(r0, c0, r1, c1))
