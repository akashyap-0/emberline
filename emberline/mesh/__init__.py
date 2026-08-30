"""Discrete-event LoRa-class mesh simulation.

Radio model
-----------
Log-distance path loss with terrain occlusion:

    RSSI = P_tx - [PL0 + 10 n log10(d/1m)] - k * max_intrusion_m

where ``n`` (~2.9) is a semi-rural exponent and the occlusion term samples
terrain along the straight line between 3 m masts and charges ``k`` dB per
metre the ridge pokes above the line-of-sight ray - a cheap stand-in for
knife-edge diffraction that still makes ridgelines matter. A link exists
when RSSI clears the configured sensitivity (LoRa SF10-class, -129 dBm).

MAC model
---------
Flooding with duplicate suppression (classic LoRa mesh), random backoff
before every transmission, and a hard per-node duty-cycle budget: each node
may transmit at most ``duty_cycle`` of wall-clock time (rolling 1 h window)
- the FCC Part 15 / ETSI-style regulatory constraint lives in config, not
code. Collisions are modelled per receiver: two packet arrivals overlapping
in time at one receiver destroy both (capture effect ignored -
conservative). Each packet also carries an independent ``packet_loss_base``
Bernoulli loss for everything the model leaves out (fading, interference).

Storm suppression: a node forwards a given packet id at most once, and
nodes that already relayed a Tier-2 alert for an incident stop relaying
lower-tier traffic for that incident; the cluster head aggregates
detections instead of every node re-flooding its own alarm.

Self-healing: nodes heartbeat; when the elected cluster head (highest
degree, health-weighted) goes silent for 2.5 heartbeat intervals, the
next-best node elects itself and announces - the event log shows the
handover.
"""

from __future__ import annotations

import heapq
import itertools
from dataclasses import dataclass
from typing import Any, Callable

import numpy as np

from ..worldgen import World


@dataclass
class Packet:
    pkt_id: int
    kind: str  # DETECT | ALERT | HEARTBEAT | HEAD_ELECT
    origin: str
    payload: dict[str, Any]
    ttl: int
    hops: int = 0


@dataclass
class MeshNode:
    node_id: str
    r: int
    c: int
    battery_v: float = 3.9
    drift_score: float = 0.0  # 0 = calibrated, 1 = fully drifted
    alive: bool = True

    @property
    def health(self) -> float:
        """0..1 corroboration weight basis: battery and drift telemetry.

        Battery below 3.3 V and accumulated baseline drift both erode trust
        in the node's confidence values (a drifting PM sensor is exactly the
        node most likely to cry wolf).
        """
        battery = float(np.clip((self.battery_v - 3.0) / 0.9, 0.0, 1.0))
        drift = float(np.clip(1.0 - self.drift_score, 0.0, 1.0))
        return battery * drift


class MeshSim:
    """Event-queue mesh simulator over a World's terrain."""

    def __init__(self, world: World, nodes: list[MeshNode], cfg: dict[str, Any],
                 rng: np.random.Generator) -> None:
        self.world = world
        self.cfg = cfg
        self.rng = rng
        self.nodes = {n.node_id: n for n in nodes}
        self.t_ms = 0.0
        self._q: list[tuple[float, int, Callable[[], None]]] = []
        self._seq = itertools.count()
        self._pkt_ids = itertools.count(1)
        self.log: list[tuple[float, str, str, str]] = []  # (t_ms, node, kind, text)
        self._seen: dict[str, set[int]] = {n: set() for n in self.nodes}
        self._airtime_used_ms: dict[str, float] = {n: 0.0 for n in self.nodes}
        self._arrivals: dict[str, list[tuple[float, float, Packet, str]]] = {
            n: [] for n in self.nodes}
        self.on_receive: Callable[[MeshNode, Packet, float], None] | None = None
        self.total_tx = 0
        self.total_delivered = 0
        self.total_collisions = 0
        self.total_cad_drops = 0
        self._busy_until_ms = 0.0  # single-collision-domain CAD approximation

        self.rssi = self._link_matrix()
        self.neighbors = {
            a: [b for b in self.nodes if b != a and self.rssi[(a, b)] >= float(cfg["sensitivity_dbm"])]
            for a in self.nodes}

    # -- radio -------------------------------------------------------------
    def _occlusion_db(self, a: MeshNode, b: MeshNode) -> float:
        """Terrain intrusion above the 3 m-mast line-of-sight ray."""
        n_samp = 24
        rs = np.linspace(a.r, b.r, n_samp)
        cs = np.linspace(a.c, b.c, n_samp)
        elev = self.world.elevation
        ground = elev[rs.astype(int), cs.astype(int)]
        mast = 3.0
        ray = np.linspace(elev[a.r, a.c] + mast, elev[b.r, b.c] + mast, n_samp)
        intrusion = float(np.clip(ground - ray, 0.0, None).max())
        return intrusion * float(self.cfg["terrain_occlusion_db_per_m"])

    def _link_matrix(self) -> dict[tuple[str, str], float]:
        out: dict[tuple[str, str], float] = {}
        cell = self.world.cell_m
        for a in self.nodes.values():
            for b in self.nodes.values():
                if a.node_id == b.node_id:
                    continue
                d = max(cell * float(np.hypot(a.r - b.r, a.c - b.c)), 1.0)
                pl = float(self.cfg["ref_loss_db"]) + 10 * float(self.cfg["path_loss_exponent"]) * np.log10(d)
                rssi = float(self.cfg["tx_power_dbm"]) - pl - self._occlusion_db(a, b)
                out[(a.node_id, b.node_id)] = rssi
        return out

    # -- event queue ---------------------------------------------------------
    def schedule(self, t_ms: float, fn: Callable[[], None]) -> None:
        heapq.heappush(self._q, (t_ms, next(self._seq), fn))

    def run_until(self, t_end_ms: float) -> None:
        while self._q and self._q[0][0] <= t_end_ms:
            t, _, fn = heapq.heappop(self._q)
            self.t_ms = max(self.t_ms, t)
            fn()
        self.t_ms = max(self.t_ms, t_end_ms)

    def log_event(self, node_id: str, kind: str, text: str) -> None:
        self.log.append((self.t_ms, node_id, kind, text))

    # -- MAC/PHY ---------------------------------------------------------------
    def duty_budget_ms(self, node_id: str) -> float:
        """Remaining allowed airtime in the (simplified: whole-sim) window."""
        horizon = max(self.t_ms, 3600_000.0)
        allowed = float(self.cfg["duty_cycle"]) * horizon
        return allowed - self._airtime_used_ms[node_id]

    def new_packet(self, kind: str, origin: str, payload: dict[str, Any], ttl: int) -> Packet:
        return Packet(next(self._pkt_ids), kind, origin, payload, ttl)

    def tx(self, src_id: str, pkt: Packet, backoff: bool = True) -> bool:
        """Queue a transmission: random backoff + CAD (listen-before-talk).

        LoRa radios support channel-activity detection; modelling the whole
        mesh as one collision domain and deferring while the channel is busy
        is conservative (it over-serialises far-apart nodes) but is what
        keeps 50 simultaneous triggers from melting into pure ALOHA loss.
        Hidden-terminal collisions are still possible when two nodes pick
        overlapping start times, and are resolved per receiver in _deliver.
        """
        src = self.nodes[src_id]
        if not src.alive:
            return False
        # Heartbeats are tiny frames; data packets pay full SF10 airtime.
        airtime = (float(self.cfg.get("heartbeat_airtime_ms", 80.0))
                   if pkt.kind == "HEARTBEAT" else float(self.cfg["airtime_ms"]))
        delay = float(self.rng.uniform(0, float(self.cfg["backoff_ms_max"]))) if backoff else 0.0
        self.schedule(self.t_ms + delay, lambda: self._attempt(src_id, pkt, airtime, 0))
        return True

    def _attempt(self, src_id: str, pkt: Packet, airtime: float, tries: int) -> None:
        src = self.nodes[src_id]
        if not src.alive:
            return
        if self.t_ms < self._busy_until_ms:  # channel busy: re-backoff
            if tries >= 8:
                self.total_cad_drops += 1
                return
            retry = self._busy_until_ms + float(self.rng.uniform(1.0, float(self.cfg["backoff_ms_max"])))
            self.schedule(retry, lambda: self._attempt(src_id, pkt, airtime, tries + 1))
            return
        if self.duty_budget_ms(src_id) < airtime:
            self.log_event(src_id, "DUTY", f"tx blocked by duty-cycle budget ({pkt.kind})")
            return
        start = self.t_ms
        self._busy_until_ms = max(self._busy_until_ms, start + airtime)
        self._airtime_used_ms[src_id] += airtime
        self.total_tx += 1
        self._propagate(src_id, pkt, start, start + airtime)

    def _propagate(self, src_id: str, pkt: Packet, start: float, end: float) -> None:
        for dst_id in self.neighbors[src_id]:
            dst = self.nodes[dst_id]
            if not dst.alive:
                continue
            if self.rng.random() < float(self.cfg["packet_loss_base"]):
                continue
            self._arrivals[dst_id].append((start, end, pkt, src_id))
            self.schedule(end, lambda d=dst_id, s=start, e=end, p=pkt, f=src_id:
                          self._deliver(d, s, e, p, f))

    def _deliver(self, dst_id: str, start: float, end: float, pkt: Packet, from_id: str) -> None:
        dst = self.nodes[dst_id]
        if not dst.alive:
            return
        # Collision: any other arrival at this receiver overlapping in time.
        overlaps = [a for a in self._arrivals[dst_id]
                    if a[2].pkt_id != pkt.pkt_id and a[0] < end and a[1] > start]
        self._arrivals[dst_id] = [a for a in self._arrivals[dst_id] if a[1] > self.t_ms - 1]
        if overlaps:
            self.total_collisions += 1
            return
        if pkt.pkt_id in self._seen[dst_id]:
            return
        self._seen[dst_id].add(pkt.pkt_id)
        self.total_delivered += 1
        if self.on_receive is not None:
            self.on_receive(dst, pkt, self.t_ms)

    # -- reporting -------------------------------------------------------------
    def channel_utilization(self) -> float:
        """Network airtime / (sim time x nodes) - the storm-suppression metric."""
        if self.t_ms <= 0:
            return 0.0
        return sum(self._airtime_used_ms.values()) / (self.t_ms * len(self.nodes))

    def format_log(self) -> str:
        lines = []
        for t, node, kind, text in self.log:
            lines.append(f"[{t / 1000:9.3f}s] {node:>4} {kind:<10} {text}")
        return "\n".join(lines)

    def kill_node(self, node_id: str) -> None:
        self.nodes[node_id].alive = False
        self.log_event(node_id, "KILL", "node power lost")
