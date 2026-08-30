"""Escalation ladder + cluster-head protocol on top of MeshSim.

Ladder (thresholds in ``config.yaml -> mesh.escalation``):

* **Tier 0 - chirp**: one node, moderate confidence. Local audible chirp
  only; a DETECT packet heads for the cluster head. Cheap to be wrong.
* **Tier 1 - voice w/ bearing**: sustained high confidence from a healthy
  node, OR health-weighted 2-node corroboration. Nodes speak a bearing
  ("smoke, bearing two-two-five").
* **Tier 2 - full cascade**: multi-node corroboration with weighted score
  over threshold. Every reachable node sirens.

Corroboration score = sum over distinct origins of (health_weight x
confidence) within the corroboration window. Health weights are clipped
below at ``health_weight_floor`` so even a degraded node can contribute a
little, but a drifting/low-battery node cannot single-handedly trigger a
cascade - that is the design answer to "what if one sensor goes insane".

Cluster head = alive node maximising degree x health (best-connected,
trusted node). Heads are re-elected when heartbeats go silent (self-heal),
and detections are aggregated at the head instead of every node flooding
its own alarm - the broadcast-storm control.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from . import MeshNode, MeshSim, Packet

DETECT_TTL = 8
ALERT_TTL = 8


@dataclass
class Detection:
    t_ms: float
    origin: str
    confidence: float
    weight: float
    bearing_deg: float


@dataclass
class Incident:
    incident_id: int
    tier: int = 0
    detections: dict[str, Detection] = field(default_factory=dict)


class MeshProtocol:
    """Application layer: detection reporting, aggregation, sirens, healing."""

    def __init__(self, sim: MeshSim, cfg: dict[str, Any]) -> None:
        self.sim = sim
        self.cfg = cfg
        e = cfg["escalation"]
        self.t0_conf = float(e["tier0_conf"])
        self.t1_conf = float(e["tier1_conf"])
        self.t1_score = float(e["tier1_score"])
        self.t2_score = float(e["tier2_score"])
        self.window_ms = float(cfg["corroboration_window_s"]) * 1000.0
        self.w_floor = float(cfg["health_weight_floor"])
        self.incident = Incident(incident_id=1)
        self.head_id = self._elect()
        self.sirens: dict[str, int] = {}  # node -> loudest tier sounded
        self.tier_times_ms: dict[int, float] = {}
        self._last_detect_tx: dict[str, float] = {}
        self._suppressed: set[str] = set()
        self._last_heartbeat: dict[str, float] = {n: 0.0 for n in sim.nodes}
        self._high_hist: dict[str, list[float]] = {}  # origin -> high-conf report times
        self._mourned: set[str] = set()  # nodes announced as lost
        sim.on_receive = self._on_receive
        sim.log_event(self.head_id, "HEAD", "elected cluster head (degree x health)")
        self._schedule_heartbeats()

    # -- election / healing --------------------------------------------------
    def _elect(self, exclude: set[str] | None = None) -> str:
        best, best_score = None, -1.0
        for nid, node in self.sim.nodes.items():
            if not node.alive or (exclude and nid in exclude):
                continue
            score = len(self.sim.neighbors[nid]) * max(node.health, 1e-3)
            if score > best_score:
                best, best_score = nid, score
        assert best is not None, "no alive nodes to elect"
        return best

    def _schedule_heartbeats(self) -> None:
        period = float(self.cfg.get("heartbeat_period_s", 60.0)) * 1000.0

        def beat(nid: str) -> None:
            node = self.sim.nodes[nid]
            if node.alive:
                pkt = self.sim.new_packet("HEARTBEAT", nid, {"health": node.health}, ttl=1)
                self.sim.tx(nid, pkt)
                self.sim.schedule(self.sim.t_ms + period, lambda: beat(nid))

        def check_head() -> None:
            # Mourn ANY node whose heartbeats stopped: flooding needs no route
            # repair, but the operator (and the demo) should see the loss.
            for nid in self.sim.nodes:
                silent = self.sim.t_ms - self._last_heartbeat[nid]
                if nid != self.head_id and nid not in self._mourned and \
                        self._last_heartbeat[nid] > 0 and silent > 2.5 * period:
                    self._mourned.add(nid)
                    self.sim.log_event(nid, "SELF-HEAL",
                                       f"{nid} heartbeats silent {silent/1000:.0f}s - "
                                       "presumed lost; mesh flooding re-routes around it")
            head = self.sim.nodes[self.head_id]
            silent_ms = self.sim.t_ms - self._last_heartbeat[self.head_id]
            if (not head.alive) or silent_ms > 2.5 * period:
                old = self.head_id
                self.head_id = self._elect(exclude={old})
                self.sim.log_event(self.head_id, "SELF-HEAL",
                                   f"cluster head {old} silent {silent_ms/1000:.0f}s -> "
                                   f"{self.head_id} self-elected")
                pkt = self.sim.new_packet("HEAD_ELECT", self.head_id,
                                          {"old": old}, ttl=ALERT_TTL)
                self.sim.tx(self.head_id, pkt)
            self.sim.schedule(self.sim.t_ms + period, check_head)

        for i, nid in enumerate(self.sim.nodes):
            self.sim.schedule(i * 700.0, lambda n=nid: beat(n))
        self.sim.schedule(period * 1.5, check_head)

    # -- detection entry point -------------------------------------------------
    def report_detection(self, node_id: str, confidence: float, bearing_deg: float) -> None:
        """Called by the sensing layer when a node's classifier fires."""
        node = self.sim.nodes[node_id]
        if not node.alive or confidence < self.t0_conf:
            return
        self.sim.log_event(node_id, "TIER-0",
                           f"local chirp: P(fire)={confidence:.2f} bearing {bearing_deg:.0f} deg")
        self.sirens[node_id] = max(self.sirens.get(node_id, -1), 0)
        self.tier_times_ms.setdefault(0, self.sim.t_ms)
        # Rate-limit own detect packets (storm control at the source).
        last = self._last_detect_tx.get(node_id, -1e12)
        if self.sim.t_ms - last < 30_000.0 or node_id in self._suppressed:
            return
        self._last_detect_tx[node_id] = self.sim.t_ms
        w = float(np.clip(node.health, self.w_floor, 1.0))
        pkt = self.sim.new_packet("DETECT", node_id,
                                  {"conf": confidence, "weight": w, "bearing": bearing_deg},
                                  ttl=DETECT_TTL)
        self.sim.tx(node_id, pkt)
        if node_id == self.head_id:  # head hears itself directly
            self._head_ingest(node_id, confidence, w, bearing_deg)

    # -- receive dispatch ------------------------------------------------------
    def _on_receive(self, node: MeshNode, pkt: Packet, t_ms: float) -> None:
        if pkt.kind == "HEARTBEAT":
            self._last_heartbeat[pkt.origin] = t_ms
            return
        if pkt.kind == "DETECT":
            if node.node_id == self.head_id:
                self._head_ingest(pkt.origin, float(pkt.payload["conf"]),
                                  float(pkt.payload["weight"]), float(pkt.payload["bearing"]))
            if pkt.ttl > 1 and node.node_id not in self._suppressed:
                fwd = Packet(pkt.pkt_id, pkt.kind, pkt.origin, pkt.payload,
                             pkt.ttl - 1, pkt.hops + 1)
                self.sim.tx(node.node_id, fwd)
            return
        if pkt.kind == "ALERT":
            tier = int(pkt.payload["tier"])
            if self.sirens.get(node.node_id, -1) < tier:
                self.sirens[node.node_id] = tier
                what = {1: "voice alert w/ bearing", 2: "FULL SIREN CASCADE"}[tier]
                self.sim.log_event(node.node_id, f"TIER-{tier}",
                                   f"{what} (hop {pkt.hops}, incident {pkt.payload['incident']})")
            if tier >= 2:
                self._suppressed.add(node.node_id)  # stop relaying lower-tier chatter
            if pkt.ttl > 1:
                fwd = Packet(pkt.pkt_id, pkt.kind, pkt.origin, pkt.payload,
                             pkt.ttl - 1, pkt.hops + 1)
                self.sim.tx(node.node_id, fwd)
            return
        if pkt.kind == "HEAD_ELECT":
            self.head_id = pkt.origin
            if pkt.ttl > 1:
                fwd = Packet(pkt.pkt_id, pkt.kind, pkt.origin, pkt.payload,
                             pkt.ttl - 1, pkt.hops + 1)
                self.sim.tx(node.node_id, fwd)
            return

    # -- ladder ---------------------------------------------------------------
    def _head_ingest(self, origin: str, conf: float, weight: float, bearing: float) -> None:
        inc = self.incident
        det = Detection(self.sim.t_ms, origin, conf, weight, bearing)
        prev = inc.detections.get(origin)
        if prev is None or conf * weight > prev.confidence * prev.weight:
            inc.detections[origin] = det
        if conf >= self.t1_conf and weight >= 0.7:
            self._high_hist.setdefault(origin, []).append(self.sim.t_ms)
        self._evaluate()

    def _sustained_single(self, now: float) -> bool:
        """>=2 high-confidence reports >=20 s apart from one healthy node.

        A single 60-s window can spike on a confounder (measured ambient FPR
        is several windows/node-day); requiring persistence before the
        single-node Tier-1 path fires keeps one hot window from voicing an
        alert across the whole town, at the cost of ~30 s extra latency.
        Multi-node corroboration paths are unaffected.
        """
        for times in self._high_hist.values():
            live = [t for t in times if now - t <= self.window_ms]
            if len(live) >= 2 and (max(live) - min(live)) >= 20_000.0:
                return True
        return False

    def _evaluate(self) -> None:
        inc = self.incident
        now = self.sim.t_ms
        live = [d for d in inc.detections.values() if now - d.t_ms <= self.window_ms]
        if not live:
            return
        score = sum(d.weight * d.confidence for d in live)
        distinct = len({d.origin for d in live if d.weight * d.confidence >= 0.3})
        target = inc.tier
        if distinct >= 2 and score >= self.t2_score:
            target = 2
        elif (distinct >= 2 and score >= self.t1_score) or self._sustained_single(now):
            target = max(target, 1)
        if target > inc.tier:
            inc.tier = target
            self.tier_times_ms[target] = now
            bearing = float(np.mean([d.bearing_deg for d in live]))
            self.sim.log_event(self.head_id, "ESCALATE",
                               f"tier {target}: score={score:.2f} from "
                               f"{distinct} node(s) {sorted(d.origin for d in live)}")
            pkt = self.sim.new_packet("ALERT", self.head_id,
                                      {"tier": target, "incident": inc.incident_id,
                                       "bearing": bearing,
                                       "origins": sorted(d.origin for d in live)},
                                      ttl=ALERT_TTL)
            self.sim.tx(self.head_id, pkt, backoff=False)
            self.sirens[self.head_id] = max(self.sirens.get(self.head_id, -1), target)
