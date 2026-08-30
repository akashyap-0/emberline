"""Stretch (a): hindcast harness — warning-minutes-gained on scenario files.

Loads hand-authored scenario YAMLs (see ``scenarios/``: ignition, wind,
start hour, and the minute the first human 911 report historically arrived),
replays the fire with virtual sensor nodes + classifier + mesh, and reports

    warning_minutes_gained = report_911_min - tier2_min

i.e. how many minutes before the first human call the siren cascade would
have sounded. The shipped scenarios are ILLUSTRATIVE hand-authored patterns,
not real fire records — the same file format pointed at adapter-fed real
terrain/fuel/timelines is what would make this number a claim.

Run: ``python -m emberline.foresight.hindcast scenarios/*.yaml``
"""

from __future__ import annotations

import argparse
import pathlib

import numpy as np
import yaml

from ..config import load_config, rng_for
from ..detect.model import SmokeCNN
from ..firesim import FireSim
from ..mesh import MeshNode, MeshSim
from ..mesh.escalation import MeshProtocol
from ..report import save_metrics, update_section
from ..sensors import baseline as sensor_baseline
from ..sensors import place_nodes
from ..sensors.events import fire_added_series
from ..sensors.plume import plume_concentration
from ..worldgen import generate_world
from ..worldgen.wind import WindModel


def replay(cfg: dict, scen: dict) -> dict:
    """Replay one scenario; return tier timings + warning minutes gained."""
    import torch

    from ..detect import cnn_ckpt_path

    world = generate_world(cfg, int(scen.get("world_id", 0)))
    w = scen.get("wind", {})
    wc = cfg["world"]["wind"]
    world.wind = WindModel(float(w.get("speed_ms", 8.0)), float(w.get("dir_deg", 225.0)),
                           float(wc["gust_sigma"]), float(wc["dir_sigma_deg"]),
                           float(wc["ou_tau_s"]), rng_for(cfg, "hindcast-wind"))

    snodes = place_nodes(world, int(cfg["sensors"]["n_nodes"]), rng_for(cfg, "demo-nodes"))
    mesh = MeshSim(world, [MeshNode(s.node_id, s.r, s.c) for s in snodes],
                   cfg["mesh"], rng_for(cfg, "hindcast-mesh"))
    proto = MeshProtocol(mesh, cfg["mesh"])

    state = torch.load(cnn_ckpt_path(cfg), weights_only=False)
    clf = SmokeCNN()
    clf.load_state_dict(state["model"])
    clf.eval()
    thr = float(cfg["mesh"]["escalation"]["tier0_conf"])

    max_min = int(scen.get("max_minutes", 60))
    T = max_min * 60
    t0_day = float(scen.get("start_hour_local", 12.0)) * 3600.0
    rng = rng_for(cfg, "hindcast-run")
    noise_cfg = cfg["sensors"]["noise"]
    noise = {s.node_id: np.stack([rng.normal(0, noise_cfg["pm25_sigma"], T),
                                  rng.normal(0, noise_cfg["voc_sigma"], T),
                                  rng.normal(0, noise_cfg["temp_sigma"], T),
                                  rng.normal(0, noise_cfg["rh_sigma"], T)])
             for s in snodes}
    added = {s.node_id: np.zeros((4, T)) for s in snodes}
    base = sensor_baseline(np.arange(T, dtype=float) + t0_day)

    truth = FireSim(world, rng_for(cfg, "hindcast-truth"))
    ig_r, ig_c = (int(v) for v in scen["ignition_cell"])
    for dr in (0, 1):
        for dc in (0, 1):
            truth.ignite(ig_r + dr, ig_c + dc)

    sensor_rc = np.array([[s.r, s.c] for s in snodes])
    pl = cfg["sensors"]["plume"]
    prev = np.zeros(len(snodes))
    step_s = 30
    for k in range(T // step_s - 2):
        truth.step()
        t_s = k * step_s
        burning = np.argwhere(truth.state == 1)
        scale = 1.0
        if len(burning) > 400:
            keep = rng.choice(len(burning), 400, replace=False)
            scale = len(burning) / 400
            burning = burning[keep]
        conc = scale * plume_concentration(burning, sensor_rc, world.wind.uv(t_s),
                                           world.cell_m, float(pl["q_fire"]),
                                           float(pl["sigma_y_coef"]), float(pl["sigma_z_coef"]))
        for i, s in enumerate(snodes):
            added[s.node_id][:, t_s : t_s + step_s] += fire_added_series(
                np.linspace(prev[i], conc[i], step_s), rng)
        prev = conc
        t_now = t_s + step_s
        if t_now < 60:  # need a full 60-s window before classifying
            continue
        for s in snodes:
            wnd = (base[:, t_now - 60 : t_now] + added[s.node_id][:, t_now - 60 : t_now]
                   + noise[s.node_id][:, t_now - 60 : t_now]).astype(np.float32)
            wnd[0] = np.clip(wnd[0], 0, None)
            wnd[3] = np.clip(wnd[3], 0, 100)
            conf = clf.confidence(wnd)
            if conf >= thr:
                _, wd = world.wind.at(t_s)
                proto.report_detection(s.node_id, conf, (np.rad2deg(wd) + 180.0) % 360.0)
        mesh.run_until(t_now * 1000.0)
        if proto.incident.tier >= 2:
            break

    tiers = {t: round(ms / 60000.0, 1) for t, ms in proto.tier_times_ms.items()}
    r911 = float(scen["report_911_min"])
    out = {"name": scen["name"], "report_911_min": r911,
           "tier0_min": tiers.get(0), "tier1_min": tiers.get(1),
           "tier2_min": tiers.get(2)}
    # None = the mesh never corroborated to a cascade: an honest miss, and
    # exactly the siting feedback a hindcast harness is for.
    out["warning_minutes_gained"] = (round(r911 - tiers[2], 1) if 2 in tiers else None)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description="Hindcast harness")
    ap.add_argument("scenarios", nargs="+", help="scenario YAML file(s)")
    args = ap.parse_args()
    cfg = load_config()
    rows = []
    fmt = lambda v: "—" if v is None else v  # noqa: E731
    for path in args.scenarios:
        scen = yaml.safe_load(pathlib.Path(path).read_text(encoding="utf-8"))
        r = replay(cfg, scen)
        rows.append(r)
        print(f"{r['name']}: tier0 {fmt(r['tier0_min'])} min, "
              f"tier2 {fmt(r['tier2_min'])} min, 911 at {r['report_911_min']} min -> "
              f"warning gained {fmt(r['warning_minutes_gained'])} min")
    save_metrics("hindcast", {"scenarios": rows})
    table = "\n".join(f"| {r['name']} | {fmt(r['tier0_min'])} | {fmt(r['tier2_min'])} | "
                      f"{r['report_911_min']} | **{fmt(r['warning_minutes_gained'])}** |"
                      for r in rows)
    update_section("hindcast", f"""
### Hindcast harness (stretch)

Regenerate: `python -m emberline.foresight.hindcast scenarios/*.yaml`. Scenario
files are **hand-authored illustrative patterns, not real fire records** (the
file format + adapters are the path to real hindcasts). Minutes from ignition:

| scenario | Tier-0 | Tier-2 cascade | first 911 report (authored) | warning minutes gained |
|---|---|---|---|---|
{table}

A "—" means the mesh never corroborated to a cascade: the harness is thus
also a **siting design tool** — it shows where the network layout would have
missed, before any hardware is planted.
""")
    print("wrote metrics/hindcast.json and REPORT.md section")


if __name__ == "__main__":
    main()
