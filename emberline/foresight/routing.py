"""Capacity-aware evacuation routing on the road graph. ADVISORY ONLY.

Method
------
1. **Blocking**: any road edge whose raster cells intersect the fire cone
   (probability over threshold, dilated by the safety margin) is removed.
2. **Demand**: each building contributes one vehicle at its access node.
3. **Assignment**: iterative capacity-aware loading (a 3-round approximation
   of user-equilibrium traffic assignment): route everyone by A* on current
   costs, accumulate per-edge load, then re-cost edges as

       cost = length_m * (1 + alpha * (load / capacity)^2)

   and repeat. The quadratic term is a BPR-style congestion function; three
   rounds is plenty at town scale and keeps replanning interactive.
4. **Fallback**: pre-planned static zone routes (no cone, no congestion)
   are always computed - if live replanning ever fails (graph split), the
   printed guidance degrades to the static plan rather than to nothing.

Everything returned is labelled ADVISORY: this is decision support for a
human incident commander, not an automatic order.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np

from ..worldgen import World

ALPHA = 0.15  # BPR congestion coefficient
CAR_SPEED_MS = 8.33  # 30 km/h evacuation crawl


@dataclass
class RoutePlan:
    """Routes per access node + summary. label is always ADVISORY."""

    routes: dict[tuple[int, int], list[tuple[int, int]]]  # access node -> node path
    exit_loads: dict[tuple[int, int], int]
    unreachable: list[tuple[int, int]] = field(default_factory=list)
    blocked_edges: int = 0
    label: str = "ADVISORY - decision support only"

    def eta_min(self, world: World, access: tuple[int, int]) -> float | None:
        path = self.routes.get(access)
        if not path or len(path) < 2:
            return None
        g = world.town.roads
        length = sum(g.edges[a, b]["length_m"] for a, b in zip(path, path[1:]))
        return length / CAR_SPEED_MS / 60.0

    def exit_of(self, access: tuple[int, int]) -> tuple[int, int] | None:
        path = self.routes.get(access)
        return path[-1] if path else None


def _demand_by_access(world: World) -> dict[tuple[int, int], int]:
    demand: dict[tuple[int, int], int] = {}
    for b in world.town.buildings:
        demand[b.access] = demand.get(b.access, 0) + 1
    return demand


def plan_evacuation(
    world: World,
    cone_mask: np.ndarray | None,
    cfg: dict[str, Any],
    rounds: int = 3,
) -> RoutePlan:
    """Route every occupied access node to its best exit, avoiding the cone."""
    g = world.town.roads.copy()
    capacity = float(cfg["foresight"]["road_capacity_vph"])
    exits = [e for e in world.town.exits]

    blocked = 0
    if cone_mask is not None:
        doomed = []
        for a, b, data in g.edges(data=True):
            if any(cone_mask[r, c] for r, c in data["cells"]):
                doomed.append((a, b))
        for a, b in doomed:
            g.remove_edge(a, b)
        blocked = len(doomed)
    safe_exits = [e for e in exits
                  if cone_mask is None or not cone_mask[e[0], e[1]]]

    demand = _demand_by_access(world)
    load: dict[frozenset, float] = {}

    def edge_cost(a, b, data) -> float:
        l = load.get(frozenset((a, b)), 0.0)
        return data["length_m"] * (1.0 + ALPHA * (l / capacity) ** 2)

    def heuristic(a, b) -> float:
        return float(np.hypot(a[0] - b[0], a[1] - b[1])) * world.cell_m

    routes: dict[tuple[int, int], list[tuple[int, int]]] = {}
    unreachable: list[tuple[int, int]] = []
    for _round in range(rounds):
        load.clear()
        routes.clear()
        unreachable.clear()
        for access, n_veh in demand.items():
            best_path, best_cost = None, np.inf
            for ex in safe_exits:
                if access not in g or ex not in g:
                    continue
                try:
                    path = nx.astar_path(g, access, ex, heuristic=heuristic,
                                         weight=edge_cost)
                except nx.NetworkXNoPath:
                    continue
                cost = sum(edge_cost(a, b, g.edges[a, b]) for a, b in zip(path, path[1:]))
                if cost < best_cost:
                    best_path, best_cost = path, cost
            if best_path is None:
                unreachable.append(access)
                continue
            routes[access] = best_path
            for a, b in zip(best_path, best_path[1:]):
                load[frozenset((a, b))] = load.get(frozenset((a, b)), 0.0) + n_veh

    exit_loads: dict[tuple[int, int], int] = {e: 0 for e in exits}
    for access, path in routes.items():
        exit_loads[path[-1]] = exit_loads.get(path[-1], 0) + demand[access]
    return RoutePlan(routes=routes, exit_loads=exit_loads, unreachable=unreachable,
                     blocked_edges=blocked)


def static_fallback_plan(world: World, cfg: dict[str, Any]) -> RoutePlan:
    """Pre-planned zone routes: no cone, no congestion - always available."""
    return plan_evacuation(world, None, cfg, rounds=1)
