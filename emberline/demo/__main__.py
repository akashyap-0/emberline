"""Emberline end-to-end scripted demo.

    python -m emberline.demo --scenario ridgeline [--seed N]
        [--wind-shift DEG] [--kill-node N3] [--fast]

A ~3-minute terminal story: quiet town -> BBQ correctly ignored -> ridge
ignition with the INTERNET DOWN -> mesh escalation ladder -> Foresight
probability cones + evacuation routing -> optional mid-run wind shift and
node kill -> CAP draft to outbox/ -> measured-metrics scoreboard.
Frames + GIF land in demo/out/. Plain ANSI, no GUI.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
import time

import imageio.v2 as imageio
import numpy as np
import torch

from ..config import load_config, repo_root, rng_for
from ..firesim import FireSim
from ..foresight import Foresight, estimate_ignition
from ..foresight.cap import draft_cap_alert
from ..foresight.routing import plan_evacuation
from ..mesh import MeshNode, MeshSim
from ..mesh.escalation import MeshProtocol
from ..report import load_metrics, save_metrics
from ..sensors import baseline as sensor_baseline
from ..sensors import place_nodes
from ..sensors.events import fire_added_series, make_confounder
from ..sensors.plume import plume_concentration
from ..worldgen import generate_world

# ---------------------------------------------------------------- ANSI ----
R, G, Y, B, M, C, DIM, BOLD, END = ("\033[91m", "\033[92m", "\033[93m", "\033[94m",
                                    "\033[95m", "\033[96m", "\033[2m", "\033[1m",
                                    "\033[0m")

SCENARIOS = {
    "ridgeline": "ignition on the ridge upwind of town; wind drives fire toward it",
    "valley": "ignition on low ground upwind of town",
}


class Narrator:
    def __init__(self, fast: bool) -> None:
        self.fast = fast
        self.t0 = time.time()

    def pause(self, s: float) -> None:
        if not self.fast:
            time.sleep(s)

    def say(self, text: str, color: str = "", pause: float = 0.0) -> None:
        print(f"{color}{text}{END}" if color else text, flush=True)
        self.pause(pause)

    def rule(self, title: str = "") -> None:
        line = "─" * 74
        if title:
            pad = max(2, 74 - len(title) - 4)
            print(f"{DIM}── {title} {'─' * pad}{END}", flush=True)
        else:
            print(f"{DIM}{line}{END}", flush=True)


class Classifier:
    """SmokeCNN if the checkpoint exists; transparent heuristic otherwise."""

    def __init__(self, cfg) -> None:
        from ..detect import cnn_ckpt_path
        from ..detect.model import SmokeCNN

        self.kind = "heuristic"
        self.threshold = float(cfg["mesh"]["escalation"]["tier0_conf"])
        p = cnn_ckpt_path(cfg)
        if p.exists():
            state = torch.load(p, weights_only=False)
            self.model = SmokeCNN()
            self.model.load_state_dict(state["model"])
            self.model.eval()
            self.kind = "1D-CNN"
        else:
            self.model = None

    def confidence(self, window: np.ndarray) -> float:
        if self.model is not None:
            return self.model.confidence(window)
        pm = window[0]
        return float(1 / (1 + np.exp(-(pm.mean() - 18.0) / 6.0)))  # crude fallback


def main() -> None:
    # Windows consoles may default to a legacy codepage; keep the box art safe.
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

    ap = argparse.ArgumentParser(description="Emberline scripted demo")
    ap.add_argument("--scenario", default="ridgeline", choices=sorted(SCENARIOS))
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--wind-shift", type=float, default=None, metavar="DEG",
                    help="swing mean wind by DEG mid-run and re-plan")
    ap.add_argument("--kill-node", default=None, metavar="N3",
                    help="kill this mesh node mid-cascade (self-heal demo)")
    ap.add_argument("--fast", action="store_true", help="skip narrative pauses")
    args = ap.parse_args()

    cfg = load_config()
    if args.seed is not None:
        cfg["seed"] = int(args.seed)
    nar = Narrator(args.fast)
    out_dir = repo_root() / cfg["demo"]["out_dir"] / "demo_frames"
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---------------------------------------------------------- setup ----
    world = generate_world(cfg, 0)
    snodes = place_nodes(world, int(cfg["sensors"]["n_nodes"]), rng_for(cfg, "demo-nodes"))
    mesh_nodes = [MeshNode(s.node_id, s.r, s.c) for s in snodes]
    # One deliberately degraded node: drifted + tired battery (health story).
    mesh_nodes[1].battery_v, mesh_nodes[1].drift_score = 3.35, 0.45
    mesh = MeshSim(world, mesh_nodes, cfg["mesh"], rng_for(cfg, "demo-mesh"))
    proto = MeshProtocol(mesh, cfg["mesh"])
    clf = Classifier(cfg)
    foresight = Foresight(cfg, world)
    rng = rng_for(cfg, "demo-run")

    total_min = 75
    T = total_min * 60
    t0_day = int(13.5 * 3600)  # 13:30 local, afternoon fire weather
    noise_cfg = cfg["sensors"]["noise"]
    noise = {s.node_id: np.stack([
        rng.normal(0, noise_cfg["pm25_sigma"], T),
        rng.normal(0, noise_cfg["voc_sigma"], T),
        rng.normal(0, noise_cfg["temp_sigma"], T),
        rng.normal(0, noise_cfg["rh_sigma"], T)]) for s in snodes}
    added = {s.node_id: np.zeros((4, T)) for s in snodes}
    base = sensor_baseline(np.arange(T, dtype=float) + t0_day)

    def window_at(node_id: str, t_s: int) -> np.ndarray:
        sl = np.s_[:, max(0, t_s - 60) : t_s]
        w = base[sl] + added[node_id][sl] + noise[node_id][sl]
        w[0] = np.clip(w[0], 0, None)
        w[3] = np.clip(w[3], 0, 100)
        return w.astype(np.float32)

    # ------------------------------------------------------- act 1 -------
    nar.rule("EMBERLINE — simulated wildfire early-warning demo")
    nar.say(f"{BOLD}Scenario '{args.scenario}': {SCENARIOS[args.scenario]}{END}")
    nar.say(f"World: {world.n}x{world.n} cells @ {world.cell_m:.0f} m | "
            f"{len(world.town.buildings)} buildings | {len(world.town.exits)} exit roads | "
            f"wind {world.wind_base_speed_ms:.0f} m/s toward {world.wind_base_dir_deg:.0f}°")
    nar.say(f"Mesh: {len(mesh_nodes)} solar LoRa nodes | cluster head {proto.head_id} | "
            f"classifier: {clf.kind}", pause=1.0)
    nar.say(f"{DIM}Everything below is synthetic simulation — no real-world claim.{END}",
            pause=1.0)

    nar.rule("13:30 — a quiet Saturday afternoon")
    speed0, dir0 = world.wind.at(0.0)
    nar.say(f"Ambient monitoring. Wind {speed0:.1f} m/s. All nodes nominal "
            f"(except {mesh_nodes[1].node_id}: battery 3.35 V, drift flag — "
            f"health weight {mesh_nodes[1].health:.2f}).", pause=1.5)

    # BBQ confounder at a ring node, minute 1, ~30 min long.
    bbq_rng = rng_for(cfg, "demo-bbq")
    bbq = make_confounder("bbq", len(snodes), bbq_rng)
    bbq_node = snodes[4].node_id
    bbq_t0, bbq_dur = 60, int(min(bbq.duration_s, 2400))
    env = bbq.envelope(4, np.arange(bbq_dur, dtype=float))
    added[bbq_node][:, bbq_t0 : bbq_t0 + bbq_dur] += env
    nar.say(f"13:31  {bbq_node}: neighbour lights a barbecue 20 m away "
            f"(PM {env[0].max():.0f} µg/m³ peak, VOC-rich).", pause=1.0)

    for t_check in (240, 420):
        conf = clf.confidence(window_at(bbq_node, t_check))
        others = max(clf.confidence(window_at(s.node_id, t_check))
                     for s in snodes if s.node_id != bbq_node)
        verdict = ("correctly ignored — non-fire signature"
                   if conf < clf.threshold else
                   "single node only — held at Tier 0, no corroboration")
        nar.say(f"13:{31 + t_check // 60:02d}  {bbq_node} P(fire)={conf:.2f} "
                f"({verdict}); all other nodes ≤ {others:.2f}", C, pause=1.2)

    # ------------------------------------------------------- act 2 -------
    ig_min = 8
    r0, c0, r1, c1 = world.town.district
    _, wdir = world.wind.at(ig_min * 60.0)
    cr, cc_ = (r0 + r1) / 2, (c0 + c1) / 2
    # Candidate ignitions in the upwind sector; ridgeline takes the highest
    # burnable ground, valley the lowest.
    cand = []
    ig_rng = rng_for(cfg, "demo-ignition")
    for _ in range(200):
        dist = ig_rng.uniform(55, 90)
        ang = wdir + np.pi + ig_rng.normal(0, 0.4)  # upwind ± jitter
        r = int(np.clip(cr + dist * np.sin(ang), 5, world.n - 6))
        c = int(np.clip(cc_ + dist * np.cos(ang), 5, world.n - 6))
        if world.fuel[r, c] in (1, 2, 3):
            cand.append((world.elevation[r, c], r, c, dist))
    cand.sort(reverse=(args.scenario == "ridgeline"))
    _, ig_r, ig_c, dist = cand[0]
    truth = FireSim(world, rng_for(cfg, "demo-truth"))
    for dr in (0, 1):
        for dc in (0, 1):
            truth.ignite(ig_r + dr, ig_c + dc)
    upwind_km = dist * world.cell_m / 1000

    nar.rule(f"13:{31 + ig_min:02d} — IGNITION (hidden from the system)")
    nar.say(f"A fire starts at grid ({ig_r},{ig_c}) — {upwind_km:.1f} km upwind of town, "
            f"elevation {world.elevation[ig_r, ig_c]:.0f} m.", Y, pause=1.0)
    nar.say(f"{BOLD}{R}┌────────────────────────────────────────────┐{END}")
    nar.say(f"{BOLD}{R}│  INTERNET: DOWN   CELL: DOWN   GRID: DOWN  │{END}")
    nar.say(f"{BOLD}{R}│  the mesh is on its own from here on       │{END}")
    nar.say(f"{BOLD}{R}└────────────────────────────────────────────┘{END}", pause=1.5)

    sensor_rc = np.array([[s.r, s.c] for s in snodes])
    pl = cfg["sensors"]["plume"]
    frames: list[np.ndarray] = []
    from .render import render_scene

    printed_log = 0
    hop_latencies: list[tuple[int, float]] = []
    tier_seen = 0
    forecasts_done = 0
    kill_at: float | None = None
    killed = False
    shift_at: float | None = None
    shifted = False
    cone = None
    cone_before = None
    cone_after = None
    plan_before = None
    plan_after = None
    ig_est = None
    cap_path = None
    events_metrics: dict = {"scenario": args.scenario, "backend": foresight.backend,
                            "ignition_sim_min": ig_min}
    chirp_counts: dict[str, int] = {}

    def flush_mesh_log() -> None:
        nonlocal printed_log
        for t_ms, node, kind, text in mesh.log[printed_log:]:
            if kind == "TIER-0":
                chirp_counts[node] = chirp_counts.get(node, 0) + 1
                if chirp_counts[node] == 3:
                    nar.say(f"  {DIM}[{node} keeps chirping; further chirps not shown]{END}")
                if chirp_counts[node] >= 3:
                    continue
            if kind in ("TIER-0", "TIER-1", "TIER-2", "ESCALATE", "SELF-HEAL",
                        "HEAD", "KILL", "DUTY"):
                col = {"TIER-0": C, "TIER-1": Y, "TIER-2": R, "ESCALATE": M,
                       "SELF-HEAL": G, "KILL": R}.get(kind, DIM)
                nar.say(f"  {DIM}[t+{t_ms/1000:7.1f}s]{END} {col}{node:>4} {kind:<9}{END} {text}")
                if kind in ("TIER-1", "TIER-2") and "hop" in text:
                    hops = int(text.split("hop ")[1].split(",")[0])
                    esc_t = proto.tier_times_ms.get(int(kind[-1]), t_ms)
                    hop_latencies.append((hops, t_ms - esc_t))
        printed_log = len(mesh.log)

    fire_snaps: dict[str, np.ndarray] = {}

    def do_forecast(t_s: float, tag: str):
        nonlocal cone, forecasts_done
        wall0 = time.perf_counter()
        cone = foresight.forecast(ig_est, t0_s=t_s)
        wall = time.perf_counter() - wall0
        forecasts_done += 1
        fire_snaps[tag] = truth.state.copy()
        nar.say(f"  Foresight: {cone.result.n_members}-member {foresight.backend} "
                f"ensemble → cones +10/+30/+60 min in {wall:.1f} s wall-clock.", B)
        events_metrics.setdefault("forecast_wall_s", []).append(round(wall, 2))
        frame = render_scene(world, truth.state, cone.result.prob[-1],
                             cone.thresholds, mesh_nodes, None,
                             f"{tag}: P(burn) cone +60 min vs truth (t={t_s/60:.0f} min)",
                             out_dir / f"frame_{len(frames):03d}.png")
        frames.append(frame)
        return cone

    def describe_plan(plan, label: str) -> None:
        loads = {f"({e[0]},{e[1]})": v for e, v in plan.exit_loads.items() if v > 0}
        nar.say(f"  {plan.label}", DIM)
        nar.say(f"  {label}: {len(plan.routes)} access points routed | "
                f"exit loads {loads} | {plan.blocked_edges} road edges cut by cone | "
                f"{len(plan.unreachable)} unreachable", G)
        sample = list(plan.routes)[: 2]
        for acc in sample:
            eta = plan.eta_min(world, acc)
            if eta is not None:
                nar.say(f"    e.g. block at {acc} → exit {plan.exit_of(acc)} "
                        f"(~{eta:.0f} min at evacuation crawl)")

    # Main loop: 30-s steps.
    step_s = 30.0
    prev_conc = np.zeros(len(snodes))
    for k in range(int((total_min - ig_min) * 60 / step_s)):
        t_s = ig_min * 60 + k * step_s  # scenario clock (s)
        truth.step()
        burning = np.argwhere(truth.state == 1)
        if len(burning) > 400:
            keep = rng.choice(len(burning), 400, replace=False)
            scale = len(burning) / 400
            burning = burning[keep]
        else:
            scale = 1.0
        conc = scale * plume_concentration(burning, sensor_rc, world.wind.uv(t_s),
                                           world.cell_m, float(pl["q_fire"]),
                                           float(pl["sigma_y_coef"]),
                                           float(pl["sigma_z_coef"]))
        for i, s in enumerate(snodes):
            seg = np.linspace(prev_conc[i], conc[i], int(step_s))
            fire_add = fire_added_series(seg, rng)
            added[s.node_id][:, int(t_s) : int(t_s + step_s)] += fire_add
        prev_conc = conc

        # Classify current windows; feed the mesh.
        t_now = int(t_s + step_s)
        for s in snodes:
            node = mesh.nodes[s.node_id]
            if not node.alive:
                continue
            conf = clf.confidence(window_at(s.node_id, t_now))
            if conf >= clf.threshold:
                _, wd = world.wind.at(t_s)
                bearing = (np.rad2deg(wd) + 180.0) % 360.0  # smoke comes from upwind
                proto.report_detection(s.node_id, conf, bearing)

        mesh.run_until((t_now - ig_min * 60) * 1000.0)
        flush_mesh_log()

        # Tier transitions drive the story.
        if proto.incident.tier > tier_seen:
            tier_seen = proto.incident.tier
            events_metrics[f"tier{tier_seen}_sim_s_after_ignition"] = t_now - ig_min * 60
            if tier_seen >= 1 and ig_est is None:
                dets = [((mesh.nodes[d.origin].r, mesh.nodes[d.origin].c), d.bearing_deg)
                        for d in proto.incident.detections.values()]
                ig_est = estimate_ignition(dets, world)
                err_m = float(np.hypot(ig_est[0] - ig_r, ig_est[1] - ig_c) * world.cell_m)
                nar.say(f"  Ignition estimate from {len(dets)} bearing(s): "
                        f"({ig_est[0]},{ig_est[1]}) — {err_m:.0f} m from truth.", B, pause=1.0)
                events_metrics["ignition_estimate_error_m"] = round(err_m)
            if tier_seen >= 2:
                if hop_latencies:
                    worst = max(h for h, _ in hop_latencies)
                    med = float(np.median([l for _, l in hop_latencies]))
                    nar.say(f"  Cascade delivery: max {worst} hops, median per-alert "
                            f"latency {med/1000:.1f} s after escalation.", M, pause=1.0)
                    events_metrics["cascade_max_hops"] = worst
                    events_metrics["cascade_median_latency_s"] = round(med / 1000, 2)
                cone_before = do_forecast(t_s, "Tier-2")
                fcfg = cfg["foresight"]
                rmask = cone.routing_mask(float(fcfg["routing_horizon_min"]),
                                          float(fcfg["routing_threshold"]))
                plan_before = plan_evacuation(world, rmask, cfg)
                describe_plan(plan_before, "Evacuation plan")
                nar.pause(1.5)
                cap_path = draft_cap_alert(
                    cfg, proto.incident.incident_id, 2, cone.union_mask(), world.cell_m,
                    f"{len(plan_before.routes)} routed access points; see posted routes.")
                if args.kill_node:
                    kill_at = t_s + 120.0
                if args.wind_shift:
                    shift_at = t_s + 600.0

        if kill_at and not killed and t_s >= kill_at:
            killed = True
            if args.kill_node in mesh.nodes:
                nar.rule(f"node failure injected: {args.kill_node}")
                was_head = args.kill_node == proto.head_id
                mesh.kill_node(args.kill_node)
                events_metrics["kill_sim_s_after_ignition"] = round(t_s - ig_min * 60)
                events_metrics["killed_node"] = args.kill_node
                nar.say(f"  {args.kill_node} destroyed mid-cascade "
                        f"({'it was the CLUSTER HEAD' if was_head else 'relay lost'}); "
                        f"watching for self-heal…", R, pause=1.0)
            else:
                nar.say(f"  --kill-node {args.kill_node}: no such node "
                        f"(have {sorted(mesh.nodes)})", Y)

        if shift_at and not shifted and t_s >= shift_at:
            shifted = True
            nar.rule(f"wind shift: {args.wind_shift:+.0f}°")
            world.wind.apply_shift(t_s, dir_delta_deg=float(args.wind_shift))
            s_now, d_now = world.wind.at(t_s + 1)
            events_metrics["windshift_sim_s_after_ignition"] = round(t_s - ig_min * 60)
            events_metrics["windshift_deg"] = float(args.wind_shift)
            nar.say(f"  Front passage: wind now {s_now:.1f} m/s toward "
                    f"{np.rad2deg(d_now) % 360:.0f}°. Re-forecasting…", Y, pause=1.0)
            cone_after = do_forecast(t_s, "Post-shift")
            fcfg = cfg["foresight"]
            plan_after = plan_evacuation(
                world, cone.routing_mask(float(fcfg["routing_horizon_min"]),
                                         float(fcfg["routing_threshold"])), cfg)
            nar.say("  BEFORE (exit loads): "
                    f"{ {f'({e[0]},{e[1]})': v for e, v in plan_before.exit_loads.items()} }")
            nar.say("  AFTER  (exit loads): "
                    f"{ {f'({e[0]},{e[1]})': v for e, v in plan_after.exit_loads.items()} }")
            moved = sum(1 for a in plan_before.routes
                        if a in plan_after.routes
                        and plan_after.routes[a][-1] != plan_before.routes[a][-1])
            nar.say(f"  Re-plan moved {moved} access points to a different exit; "
                    f"{plan_after.blocked_edges} edges now cut (was "
                    f"{plan_before.blocked_edges}).", G, pause=1.5)
            describe_plan(plan_after, "Updated plan")
            events_metrics["replan_moved_access_points"] = moved

        # Periodic truth frame for the GIF.
        if k % 10 == 9:
            frame = render_scene(world, truth.state,
                                 cone.result.prob[-1] if cone else None,
                                 cone.thresholds if cone else [],
                                 mesh_nodes, None,
                                 f"t = {t_s/60:.0f} min — truth vs standing cone",
                                 out_dir / f"frame_{len(frames):03d}.png")
            frames.append(frame)

    flush_mesh_log()

    # First self-heal announcement after the injected kill (timeline asset).
    if killed:
        k_s = float(events_metrics.get("kill_sim_s_after_ignition", 0))
        heals = [t for t, _, k, _ in mesh.log if k == "SELF-HEAL" and t / 1000.0 >= k_s]
        if heals:
            events_metrics["selfheal_sim_s_after_ignition"] = round(heals[0] / 1000.0)

    # ---- raw artifact dump for pitch-quality re-rendering (Phase 12) ----
    # Everything demo/pitch_assets.py draws comes from THIS run, via these
    # files — no numbers are ever re-invented at plot time.
    arts_dir = out_dir.parent
    npz: dict[str, np.ndarray] = {
        "fire_state_end": truth.state.astype(np.uint8),
        "ignition_truth": np.array([ig_r, ig_c]),
        "node_rc": np.array([[n.r, n.c] for n in mesh_nodes]),
        "node_alive": np.array([n.alive for n in mesh_nodes]),
        "node_health": np.array([n.health for n in mesh_nodes]),
    }
    if ig_est is not None:
        npz["ignition_estimate"] = np.array(ig_est)
    if cone_before is not None:
        npz["cone_before_prob"] = cone_before.result.prob.astype(np.float32)
        npz["cone_horizons_min"] = np.array(cone_before.result.horizons_min)
        npz["cone_thresholds"] = np.array(cone_before.thresholds)
        fcfg = cfg["foresight"]
        npz["routing_mask_before"] = cone_before.routing_mask(
            float(fcfg["routing_horizon_min"]), float(fcfg["routing_threshold"]))
    if cone_after is not None:
        npz["cone_after_prob"] = cone_after.result.prob.astype(np.float32)
        fcfg = cfg["foresight"]
        npz["routing_mask_after"] = cone_after.routing_mask(
            float(fcfg["routing_horizon_min"]), float(fcfg["routing_threshold"]))
    if "Tier-2" in fire_snaps:
        npz["fire_state_tier2"] = fire_snaps["Tier-2"].astype(np.uint8)
    if "Post-shift" in fire_snaps:
        npz["fire_state_postshift"] = fire_snaps["Post-shift"].astype(np.uint8)
    t2 = events_metrics.get("tier2_sim_s_after_ignition")
    if t2 is not None:
        npz["wind_uv_at_tier2"] = np.array(world.wind.uv(ig_min * 60 + float(t2)))
    if shifted:
        ts = float(events_metrics["windshift_sim_s_after_ignition"])
        npz["wind_uv_after_shift"] = np.array(world.wind.uv(ig_min * 60 + ts + 60.0))
    np.savez_compressed(arts_dir / "last_run_artifacts.npz", **npz)

    def plan_json(plan) -> dict | None:
        if plan is None:
            return None
        return {"routes": {f"{r},{c}": [[int(pr), int(pc)] for pr, pc in path]
                           for (r, c), path in plan.routes.items()},
                "exit_loads": {f"{r},{c}": int(v) for (r, c), v in plan.exit_loads.items()},
                "blocked_edges": int(plan.blocked_edges),
                "unreachable": len(plan.unreachable), "label": plan.label}

    (arts_dir / "last_run_artifacts.json").write_text(json.dumps({
        "plan_before": plan_json(plan_before),
        "plan_after": plan_json(plan_after),
        "mesh_log": [[t, n, k, txt] for t, n, k, txt in mesh.log],
        "neighbors": {k: list(v) for k, v in mesh.neighbors.items()},
        "head_final": proto.head_id,
        "args": {"scenario": args.scenario, "wind_shift": args.wind_shift,
                 "kill_node": args.kill_node},
    }, indent=1), encoding="utf-8")

    # ------------------------------------------------------- act 6 -------
    if cap_path:
        nar.rule("CAP draft → outbox/ (awaiting human approval)")
        doc = json.loads(pathlib.Path(cap_path).read_text(encoding="utf-8"))
        brief = {k: doc[k] for k in ("identifier", "status", "msgType")}
        brief["info.headline"] = doc["info"]["headline"]
        brief["info.instruction"] = doc["info"]["instruction"][:110] + "…"
        brief["polygon_vertices"] = len(doc["info"]["area"]["polygon"])
        nar.say(json.dumps(brief, indent=2))
        nar.say(f"  full draft: {cap_path}", DIM, pause=1.5)

    gif = out_dir.parent / "demo.gif"
    if frames:
        imageio.mimsave(gif, frames, fps=2)
    nar.say(f"\nSaved {len(frames)} frames → {out_dir} and {gif}", G)

    for tier, t_ms in proto.tier_times_ms.items():
        events_metrics.setdefault(f"tier{tier}_sim_s_after_ignition", round(t_ms / 1000.0))
    burned_ha = float((truth.state > 0).sum()) * (world.cell_m**2) / 10_000
    events_metrics["truth_burned_ha_at_end"] = round(burned_ha, 1)
    events_metrics["channel_utilization"] = round(mesh.channel_utilization(), 4)
    events_metrics["mesh_tx_total"] = mesh.total_tx
    events_metrics["mesh_collisions"] = mesh.total_collisions
    save_metrics("demo_last_run", events_metrics)

    nar.rule("SCOREBOARD — every number measured by code in this repo")
    sur, det = load_metrics("surrogate"), load_metrics("detect")
    rows: list[tuple[str, str]] = []
    if sur:
        v = sur["val"]
        rows += [("Surrogate IoU +10/+30/+60 (val worlds)",
                  f"{v['iou_10']:.3f} / {v['iou_30']:.3f} / {v['iou_60']:.3f}"),
                 ("  held-out wind regime IoU@+30", f"{sur['holdout_wind_regime'].get('iou_30', float('nan')):.3f}"),
                 ("  worst-case (p5) IoU@+30", f"{sur['worst_case_p5_iou_30']:.3f}"),
                 ("  fire-arrival MAE", f"{v['arrival_mae_min']:.1f} min"),
                 ("  ensemble speedup vs physics", f"{sur['speedup']['speedup_x']:.1f}×")]
    if det:
        rows += [("Detector F1 (CNN vs GBM, val events)",
                  f"{det['cnn']['f1']:.3f} vs {det['gbm']['f1']:.3f}"),
                 ("  ambient false positives", f"{det['ambient_day']['fp_per_node_day_cnn']:.1f} /node-day"),
                 ("  detection latency (median)", f"{det['latency']['latency_median_s']:.0f} s")]
    for key, label in [("tier0_sim_s_after_ignition", "This run: ignition → Tier-0"),
                       ("tier1_sim_s_after_ignition", "This run: ignition → Tier-1"),
                       ("tier2_sim_s_after_ignition", "This run: ignition → Tier-2 cascade")]:
        if key in events_metrics:
            rows.append((label, f"{events_metrics[key]:.0f} s"))
    if "ignition_estimate_error_m" in events_metrics:
        rows.append(("This run: ignition estimate error", f"{events_metrics['ignition_estimate_error_m']} m"))
    rows += [("This run: mesh channel utilization", f"{events_metrics['channel_utilization']*100:.2f} %"),
             ("This run: truth burned area at t=75 min", f"{burned_ha:.0f} ha")]
    width = max(len(a) for a, _ in rows) + 2
    for a, b in rows:
        nar.say(f"  {a:<{width}} {BOLD}{b}{END}")
    nar.say(f"\n{DIM}Synthetic simulation. Metrics regenerable via "
            f"emberline.surrogate.eval / emberline.detect.eval. "
            f"Wall time {time.time() - nar.t0:.0f} s.{END}")


if __name__ == "__main__":
    main()
