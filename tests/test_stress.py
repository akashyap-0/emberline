"""Phase 13 stress tests: multi-fire, mass node failure, in-town ignition,
all-degraded-mesh corroboration. Every assertion here encodes MEASURED system
behaviour — including designed limitations (single-incident conflation, the
mass-corroboration escape hatch) that REPORT.md's stress section documents.
"""

from __future__ import annotations

import pytest
from scipy import ndimage

from emberline.config import load_config, rng_for
from emberline.firesim import FireSim
from emberline.mesh import MeshNode, MeshSim
from emberline.mesh.escalation import MeshProtocol
from emberline.sensors import place_nodes
from emberline.worldgen import generate_world


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def world(cfg):
    return generate_world(cfg, 0)


def _demo_mesh(cfg, world, degrade_all: bool = False):
    snodes = place_nodes(world, int(cfg["sensors"]["n_nodes"]), rng_for(cfg, "demo-nodes"))
    nodes = [MeshNode(s.node_id, s.r, s.c) for s in snodes]
    if degrade_all:
        for n in nodes:
            n.battery_v, n.drift_score = 3.15, 0.7  # health < w_floor for every node
    mesh = MeshSim(world, nodes, cfg["mesh"], rng_for(cfg, "stress-mesh"))
    return mesh, MeshProtocol(mesh, cfg["mesh"])


# ---------------------------------------------------------------- (a) -------
def test_two_simultaneous_ignitions_burn_as_two_fronts(cfg, world):
    sim = FireSim(world, rng_for(cfg, "stress-2fires"))
    for r, c in ((154, 138), (185, 40)):  # ridge SE + valley SW, ~1 km apart
        for dr in (0, 1):
            for dc in (0, 1):
                sim.ignite(r + dr, c + dc)
    sim.run(15.0)
    touched = sim.state > 0
    labels, n_blobs = ndimage.label(touched)
    assert n_blobs >= 2, "two distant ignitions must burn as separate fronts"
    for r, c in ((154, 138), (185, 40)):
        blob = labels[r, c]
        assert blob > 0 and (labels == blob).sum() >= 8, \
            f"fire at ({r},{c}) failed to establish"


def test_two_fire_clusters_conflate_and_post_cascade_reports_are_suppressed(cfg, world):
    """MEASURED LIMITATIONS under test (documented in REPORT.md stress section):

    1. Detections from two physically distinct fires corroborate ONE incident
       (there is no bearing clustering), so the incident's mean bearing points
       between the fires.
    2. Once the Tier-2 cascade floods, storm suppression drops later DETECT
       packets at the source — a second fire reported after the first cascade
       never reaches the head, so Foresight is never pointed at it.
    """
    mesh, proto = _demo_mesh(cfg, world)
    # Fire A (east, bearing ~90) and fire B (west, ~270) report pre-cascade.
    mesh.schedule(1000.0, lambda: proto.report_detection("N0", 0.95, 90.0))
    mesh.schedule(1200.0, lambda: proto.report_detection("N3", 0.94, 270.0))
    mesh.run_until(240_000.0)
    assert proto.incident.tier == 2, "cross-fire corroboration still cascades"
    origins = set(proto.incident.detections)
    assert {"N0", "N3"} <= origins, "pre-cascade reports from both fires ingest"
    assert proto.incident.incident_id == 1, "single incident for two fires"

    # Post-cascade: fresh nodes smelling either fire are suppressed at source.
    t0 = mesh.t_ms
    mesh.schedule(t0 + 1000.0, lambda: proto.report_detection("N8", 0.93, 95.0))
    mesh.schedule(t0 + 2000.0, lambda: proto.report_detection("N4", 0.92, 265.0))
    mesh.run_until(t0 + 120_000.0)
    assert "N8" not in proto.incident.detections
    assert "N4" not in proto.incident.detections


# ---------------------------------------------------------------- (b) -------
def test_majority_node_failure_mid_scenario(cfg, world):
    """Kill 7 of 12 nodes (including the cluster head) mid-run: the survivors
    must self-heal (new head), still corroborate to Tier 2, and deliver the
    cascade to every alive node."""
    mesh, proto = _demo_mesh(cfg, world)
    first_head = proto.head_id
    victims = [first_head] + [n for n in ("N9", "N10", "N11", "N1", "N5", "N6")
                              if n != first_head][:6]
    assert len(victims) == 7

    def kill_all():
        for v in victims:
            mesh.kill_node(v)

    mesh.schedule(60_000.0, kill_all)
    mesh.run_until(400_000.0)  # heartbeat mourning + head re-election window
    assert proto.head_id != first_head, "dead head must be replaced"
    assert mesh.nodes[proto.head_id].alive
    assert any(k == "SELF-HEAL" for _, _, k, _ in mesh.log)

    survivors = [nid for nid, n in mesh.nodes.items() if n.alive]
    assert len(survivors) == 5
    t0 = mesh.t_ms
    mesh.schedule(t0 + 1_000.0, lambda: proto.report_detection(survivors[0], 0.95, 220.0))
    mesh.schedule(t0 + 15_000.0, lambda: proto.report_detection(survivors[1], 0.93, 225.0))
    mesh.run_until(t0 + 240_000.0)
    assert proto.incident.tier == 2, "5-node rump mesh must still cascade"
    sirened = {nid for nid, tier in proto.sirens.items()
               if tier >= 2 and mesh.nodes[nid].alive}
    assert sirened == set(survivors), \
        f"cascade must reach every survivor (missed {set(survivors) - sirened})"


# ---------------------------------------------------------------- (c) -------
def test_ignition_inside_town_spreads_on_urban_fuel(cfg, world):
    from emberline.worldgen.fuel import URBAN

    r, c = 150, 80  # urban cell inside the town district
    assert world.fuel[r, c] == URBAN
    sim = FireSim(world, rng_for(cfg, "stress-town"))
    for dr in (0, 1):
        for dc in (0, 1):
            sim.ignite(r + dr, c + dc)
    assert (sim.state == 1).sum() == 4, "urban cells must accept forced ignition"
    sim.run(30.0)
    touched = int((sim.state > 0).sum())
    assert touched > 4, "an in-town fire must spread beyond the ignition patch"
    # Urban fuel factor (0.12) + ignition gate must keep it far slower than the
    # wildland ridge fire under the same wind (sanity, not a physics claim).
    wild = FireSim(world, rng_for(cfg, "stress-town-wild"))
    for dr in (0, 1):
        for dc in (0, 1):
            wild.ignite(154 + dr, 138 + dc)
    wild.run(30.0)
    assert touched < int((wild.state > 0).sum()), \
        "urban spread should be slower than timber-ridge spread"


# ---------------------------------------------------------------- (d) -------
def test_tier2_refuses_small_degraded_only_corroboration(cfg, world):
    """3 low-health nodes screaming at 0.99 must NOT cascade: weights clip to
    the 0.2 floor, so score 3 x 0.2 x 0.99 = 0.59 misses tier1_score (1.2),
    and no floor-weighted report clears the distinct-origin 0.3 gate."""
    mesh, proto = _demo_mesh(cfg, world, degrade_all=True)
    assert all(n.health < 0.3 for n in mesh.nodes.values())
    for i, nid in enumerate(("N0", "N7", "N8")):
        mesh.schedule(1000.0 + 2000.0 * i, lambda n=nid: proto.report_detection(n, 0.99, 90.0))
    mesh.run_until(240_000.0)
    assert proto.incident.tier < 2, \
        "a cascade must not fire from a handful of unreliable nodes"
    assert proto.incident.tier < 1, \
        "even Tier-1 needs healthy-node weight (floor-weight score stays low)"


def test_all_degraded_mesh_chirps_but_never_cascades(cfg, world):
    """MEASURED (stricter than the module docstring suggests): escalation's
    ``distinct`` count only admits detections with weight x confidence >= 0.3,
    and floor-weighted reports max out at 0.2 x 0.99 = 0.198 — so a fully
    degraded mesh can NEVER corroborate to Tier 1/2, no matter how many nodes
    scream. Local Tier-0 chirps still sound. The flip side (an all-degraded
    mesh is town-siren-deaf even for a real fire) is documented in REPORT.md.
    """
    mesh, proto = _demo_mesh(cfg, world, degrade_all=True)
    for i, nid in enumerate(list(mesh.nodes)):
        mesh.schedule(1000.0 + 1500.0 * i, lambda n=nid: proto.report_detection(n, 0.99, 90.0))
    mesh.run_until(300_000.0)
    assert 0 in proto.tier_times_ms, "local chirps must still sound"
    assert proto.incident.tier == 0, \
        "12 floor-weighted reports must not clear the distinct>=2 @ 0.3 gate"
