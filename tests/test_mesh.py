"""Phase 5 tests: multi-hop latency, storm suppression, self-heal, duty cycle."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from emberline.config import load_config, rng_for
from emberline.mesh import MeshNode, MeshSim
from emberline.mesh.escalation import MeshProtocol
from emberline.worldgen import generate_world


@pytest.fixture(scope="module")
def cfg():
    return load_config()


@pytest.fixture(scope="module")
def flat_world(cfg):
    w = generate_world(cfg, 0)
    w.elevation = np.zeros_like(w.elevation)  # no occlusion for topology tests
    return w


def _line_nodes(n: int, spacing_cells: int) -> list[MeshNode]:
    return [MeshNode(f"L{i}", 128, 10 + i * spacing_cells) for i in range(n)]


def _dense_cfg(cfg):
    """Mesh config with a dense-canopy path-loss exponent (4.5) so multi-hop
    topologies fit inside the 2.56 km world; open-terrain LoRa at n=2.9
    reaches ~5.7 km, which would make every node one hop from every other."""
    m = copy.deepcopy(cfg["mesh"])
    m["path_loss_exponent"] = 4.5
    return m


def _max_1hop_cells(mesh_cfg) -> int:
    """Cells at which RSSI ~ sensitivity for the configured link budget."""
    m = mesh_cfg
    link_db = m["tx_power_dbm"] - m["sensitivity_dbm"] - m["ref_loss_db"]
    d = 10 ** (link_db / (10 * m["path_loss_exponent"]))
    return int(d / 10.0)


def test_link_budget_sane(cfg):
    d = _max_1hop_cells(cfg["mesh"])
    assert 30 <= d <= 2000, f"1-hop range {d*10} m looks wrong for LoRa"


def test_propagation_latency_six_hops(cfg, flat_world):
    mesh_cfg = _dense_cfg(cfg)
    spacing = int(_max_1hop_cells(mesh_cfg) * 0.8)
    nodes = _line_nodes(7, spacing)
    sim = MeshSim(flat_world, nodes, mesh_cfg, rng_for(cfg, "t-hops"))
    # Only adjacent nodes should be in range.
    assert sim.neighbors["L0"] == ["L1"]
    assert set(sim.neighbors["L3"]) == {"L2", "L4"}

    proto = MeshProtocol(sim, mesh_cfg)
    got: dict[str, tuple[float, int]] = {}
    orig_rx = sim.on_receive

    def spy(node, pkt, t_ms):
        if pkt.kind == "DETECT" and node.node_id not in got:
            got[node.node_id] = (t_ms, pkt.hops)
        orig_rx(node, pkt, t_ms)

    sim.on_receive = spy
    sim.schedule(1000.0, lambda: proto.report_detection("L0", 0.95, 270.0))
    sim.run_until(60_000.0)

    assert "L6" in got, "detection must flood across the whole line"
    t6, hops6 = got["L6"]
    assert hops6 >= 5, f"L6 reached in {hops6} hops; expected a >=6-hop relay chain"
    latency_s = (t6 - 1000.0) / 1000.0
    assert latency_s < 30.0, f"6-hop propagation took {latency_s:.1f}s"


def test_storm_suppression_50_triggers(cfg, flat_world):
    rng = rng_for(cfg, "t-storm")
    nodes = [MeshNode(f"S{i}", int(100 + 15 * (i // 8)), int(60 + 15 * (i % 8)))
             for i in range(50)]
    sim = MeshSim(flat_world, nodes, cfg["mesh"], rng)
    proto = MeshProtocol(sim, cfg["mesh"])
    for i in range(50):
        sim.schedule(1000.0 + i * 5.0,
                     lambda n=f"S{i}": proto.report_detection(n, 0.9, 200.0))
    sim.run_until(180_000.0)
    util = sim.channel_utilization()
    # Naive flood: 50 origins x 50 relays; suppression must cut that hard.
    assert sim.total_tx < 50 * 50 / 2, f"{sim.total_tx} transmissions is a storm"
    assert util < 0.05, f"channel utilization {util:.3f} too high"
    assert proto.incident.tier == 2, "mass corroboration must reach Tier 2"


def test_duty_cycle_budget_respected(cfg, flat_world):
    nodes = _line_nodes(3, 20)
    sim = MeshSim(flat_world, nodes, cfg["mesh"], rng_for(cfg, "t-duty"))
    MeshProtocol(sim, cfg["mesh"])
    sim.run_until(3600_000.0)
    for nid in sim.nodes:
        used = sim._airtime_used_ms[nid]
        allowed = cfg["mesh"]["duty_cycle"] * sim.t_ms
        assert used <= allowed + 1e-6, f"{nid} used {used}ms of {allowed}ms budget"


def test_self_heal_after_head_death(cfg, flat_world):
    nodes = [MeshNode(f"H{i}", 120 + 10 * (i // 3), 100 + 10 * (i % 3)) for i in range(9)]
    sim = MeshSim(flat_world, nodes, cfg["mesh"], rng_for(cfg, "t-heal"))
    proto = MeshProtocol(sim, cfg["mesh"])
    first_head = proto.head_id
    sim.schedule(30_000.0, lambda: sim.kill_node(first_head))
    sim.run_until(400_000.0)
    assert proto.head_id != first_head, "a new head must self-elect"
    assert any(k == "SELF-HEAL" for _, _, k, _ in sim.log)
    # The mesh still escalates after healing (reroute around the dead head).
    alive = [n for n in ("H0", "H1", "H2") if n != first_head and sim.nodes[n].alive]
    sim.schedule(sim.t_ms + 1000, lambda: proto.report_detection(alive[0], 0.95, 10.0))
    sim.schedule(sim.t_ms + 2000, lambda: proto.report_detection("H8", 0.92, 15.0))
    sim.run_until(sim.t_ms + 120_000.0)
    assert proto.incident.tier >= 1


def test_rerouting_after_relay_death(cfg, flat_world):
    """Diamond topology: A-B-C and A-D-C; killing B must not cut C off."""
    mesh_cfg = _dense_cfg(cfg)
    spacing = int(_max_1hop_cells(mesh_cfg) * 0.8)
    a = MeshNode("A", 128, 30)
    b = MeshNode("B", 128 - spacing // 2, 30 + spacing)
    d = MeshNode("D", 128 + spacing // 2, 30 + spacing)
    c = MeshNode("C", 128, 30 + 2 * spacing)
    sim = MeshSim(flat_world, [a, b, d, c], mesh_cfg, rng_for(cfg, "t-reroute"))
    assert "C" not in sim.neighbors["A"], "A-C must need a relay"
    assert {"B", "D"} <= set(sim.neighbors["A"])
    proto = MeshProtocol(sim, mesh_cfg)
    sim.kill_node("B")
    got = []
    orig_rx = sim.on_receive

    def spy(node, pkt, t_ms):
        if pkt.kind == "DETECT" and node.node_id == "C":
            got.append(t_ms)
        orig_rx(node, pkt, t_ms)

    sim.on_receive = spy
    sim.schedule(1000.0, lambda: proto.report_detection("A", 0.9, 0.0))
    sim.run_until(30_000.0)
    assert got, "packet must reroute via D after B dies"


def test_health_weight_downweights_degraded_nodes(cfg, flat_world):
    nodes = [MeshNode("G1", 120, 100), MeshNode("G2", 120, 110),
             MeshNode("BAD", 120, 120, battery_v=3.15, drift_score=0.7)]
    assert nodes[2].health < 0.3 < nodes[0].health
    sim = MeshSim(flat_world, nodes, cfg["mesh"], rng_for(cfg, "t-health"))
    proto = MeshProtocol(sim, cfg["mesh"])
    # Two degraded-quality reports from BAD must not reach Tier 2 alone.
    sim.schedule(1000.0, lambda: proto.report_detection("BAD", 0.99, 0.0))
    sim.run_until(120_000.0)
    assert proto.incident.tier < 2


def test_log_is_human_readable(cfg, flat_world):
    nodes = _line_nodes(3, 20)
    sim = MeshSim(flat_world, nodes, cfg["mesh"], rng_for(cfg, "t-log"))
    proto = MeshProtocol(sim, cfg["mesh"])
    sim.schedule(500.0, lambda: proto.report_detection("L0", 0.9, 45.0))
    sim.run_until(10_000.0)
    text = sim.format_log()
    assert "TIER-0" in text and "chirp" in text and "s]" in text
