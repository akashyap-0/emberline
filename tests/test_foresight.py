"""Phase 6 tests: ignition triangulation, cone masks, routing re-plan, CAP."""

from __future__ import annotations

import json

import numpy as np
import pytest

from emberline.config import load_config
from emberline.firesim.ensemble import EnsembleResult
from emberline.foresight import Cone, estimate_ignition
from emberline.foresight.cap import draft_cap_alert, validate_cap
from emberline.foresight.routing import plan_evacuation, static_fallback_plan
from emberline.worldgen import generate_world


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def world(cfg):
    return generate_world(cfg, 0)


def test_ignition_triangulation(world):
    # Two nodes with bearings that intersect at (150, 150).
    det = [((150, 100), 0.0), ((100, 150), 90.0)]  # east from A, north from B
    r, c = estimate_ignition(det, world)
    assert abs(r - 150) <= 2 and abs(c - 150) <= 2


def test_ignition_single_bearing(world):
    r, c = estimate_ignition([((100, 100), 0.0)], world)
    assert r == 100 and c > 100, "single bearing projects up the bearing"


def _fake_cone(world, blob_rc, radius=10) -> Cone:
    n = world.n
    prob = np.zeros((3, n, n))
    rr, cc = np.ogrid[:n, :n]
    for i, grow in enumerate((0.5, 1.0, 1.5)):
        prob[i][(rr - blob_rc[0]) ** 2 + (cc - blob_rc[1]) ** 2 < (radius * grow) ** 2] = 0.9
    res = EnsembleResult(horizons_min=[10, 30, 60], prob=prob, n_members=1)
    return Cone(result=res, thresholds=[0.1, 0.3, 0.6], margin_cells=3)


def test_cone_masks_monotone(world):
    cone = _fake_cone(world, (128, 128))
    m10, m60 = cone.mask(0), cone.mask(2)
    assert m60.sum() >= m10.sum()
    assert cone.union_mask().sum() >= m60.sum()


def test_routing_avoids_cone_and_replans(cfg, world):
    baseline = static_fallback_plan(world, cfg)
    assert baseline.routes, "static plan must exist"
    assert not baseline.unreachable, "every occupied access node must reach an exit"

    # Find the busiest exit and drop a cone right on its road, mid-route.
    primary_exit = max(baseline.exit_loads, key=baseline.exit_loads.get)
    g = world.town.roads
    exit_edges = [(a, b, d) for a, b, d in g.edges(data=True)
                  if d["kind"] == "exit" and primary_exit in (a, b)]
    assert exit_edges
    cells = exit_edges[0][2]["cells"]
    mid = cells[len(cells) // 2]
    cone = _fake_cone(world, mid, radius=8)

    replan = plan_evacuation(world, cone.union_mask(), cfg)
    assert replan.blocked_edges > 0, "the cone must cut at least one road edge"
    for access, path in replan.routes.items():
        assert path[-1] != primary_exit or not cone.union_mask()[primary_exit], \
            "no route may use the cut primary exit"
        for a, b in zip(path, path[1:]):
            assert not any(cone.union_mask()[r, c] for r, c in g.edges[a, b]["cells"]), \
                "replanned route passes through the cone"
    # Most of town must still get out (the cone can strand a few nodes).
    assert len(replan.routes) >= 0.7 * len(baseline.routes)
    assert "ADVISORY" in replan.label


def test_capacity_spreads_load(cfg, world):
    plan = static_fallback_plan(world, cfg)
    used_exits = [e for e, l in plan.exit_loads.items() if l > 0]
    assert len(used_exits) >= 2, "capacity-aware assignment should use multiple exits"


def test_cap_draft_validates(cfg, world, tmp_path):
    cone = _fake_cone(world, (100, 100))
    path = draft_cap_alert(cfg, incident_id=1, tier=2, cone_mask=cone.union_mask(),
                           cell_m=world.cell_m, routes_note="Use exits E1/E2.",
                           out_dir=tmp_path)
    doc = json.loads(path.read_text(encoding="utf-8"))
    assert validate_cap(doc) == []
    assert doc["status"] == "Exercise", "synthetic system must never claim Actual"
    assert doc["info"]["area"]["polygon"][0] == doc["info"]["area"]["polygon"][-1]
    assert "ADVISORY" in doc["note"] or "ADVISORY" in doc["info"]["instruction"]


def test_cap_validator_catches_bad_docs():
    assert validate_cap({}) != []
    assert any("polygon" in e for e in validate_cap({
        "identifier": "x", "sender": "s", "sent": "t", "status": "Exercise",
        "msgType": "Alert", "scope": "Public",
        "info": {"category": "Fire", "event": "e", "urgency": "u", "severity": "s",
                 "certainty": "c", "headline": "h", "description": "d",
                 "instruction": "i", "area": {"areaDesc": "a", "polygon": [[0, 0]],
                                              "crs": "local"}}}))
