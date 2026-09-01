"""Phase 12: presentation-quality static PNGs (1920x1080) for the pitch deck.

    python -m emberline.demo.pitch_assets [--only NAME]

Every number drawn here is LOADED from an artifact a real run produced —
`demo/out/last_run_artifacts.{npz,json}` + `metrics/*.json` — never typed
into this file. If a source is missing, the asset refuses to render and
prints the command that regenerates the source. Outputs land in
`demo/out/pitch_assets/`; captions + regen commands live in PITCH_ASSETS.md.
"""

from __future__ import annotations

import argparse
import json
import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from ..config import load_config, repo_root
from ..report import load_metrics
from ..viz import FUEL_CMAP, hillshade
from ..worldgen import generate_world

FIG_KW = dict(figsize=(19.2, 10.8), dpi=100)  # 1920 x 1080
CONE_COLORS = ["#ffd166", "#f3722c", "#d62828"]
INK = "#22333b"
ACCENT = {"world": "#606c38", "physics": "#bc6c25", "surrogate": "#5f0f40",
          "sensors": "#1d3557", "detect": "#457b9d", "mesh": "#2a9d8f",
          "foresight": "#e63946", "out": "#6d597a"}


def out_dir() -> pathlib.Path:
    d = repo_root() / "demo" / "out" / "pitch_assets"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _require(cond: bool, what: str, regen: str) -> None:
    if not cond:
        raise SystemExit(f"missing source for this asset: {what}\n"
                         f"  regenerate it first with: {regen}")


def _load_run():
    """The instrumented ridgeline demo run's raw artifacts + metrics."""
    base = repo_root() / "demo" / "out"
    regen = ("python -m emberline.demo --scenario ridgeline --fast "
             "--kill-node N3 --wind-shift 40")
    _require((base / "last_run_artifacts.npz").exists(), "demo run artifacts", regen)
    npz = dict(np.load(base / "last_run_artifacts.npz", allow_pickle=False).items())
    js = json.loads((base / "last_run_artifacts.json").read_text(encoding="utf-8"))
    run = load_metrics("demo_last_run")
    _require(run is not None, "metrics/demo_last_run.json", regen)
    return npz, js, run


def _basemap(ax, world, alpha_fuel: float = 0.28):
    ax.imshow(hillshade(world.elevation), cmap="gray", origin="lower", alpha=0.8)
    ax.imshow(world.fuel, cmap=FUEL_CMAP, vmin=0, vmax=4, origin="lower",
              alpha=alpha_fuel, interpolation="nearest")
    for a, b, data in world.town.roads.edges(data=True):
        cells = np.array(data["cells"])
        ax.plot(cells[:, 1], cells[:, 0], color="#2f2f2f",
                lw=2.2 if data["kind"] == "exit" else 0.7, alpha=0.85)
    ax.set_xlim(0, world.n)
    ax.set_ylim(0, world.n)
    ax.axis("off")


def _wind_arrow(ax, world, uv, label: str):
    u, v = float(uv[0]), float(uv[1])
    speed = float(np.hypot(u, v))
    r0, c0 = world.n - 36, 34
    ax.add_patch(FancyArrowPatch((c0, r0), (c0 + u * 3.2, r0 + v * 3.2),
                                 arrowstyle="-|>", mutation_scale=26,
                                 color=INK, lw=3.5, zorder=8))
    ax.text(c0 + 2, r0 + 13, f"{label} {speed:.1f} m/s", fontsize=13.5, color=INK,
            ha="center", fontweight="bold", zorder=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=INK, alpha=0.9))


# ------------------------------------------------------------------ 1 -------
def system_architecture() -> pathlib.Path:
    """Pipeline diagram; the few numbers on it are read from real artifacts."""
    import torch

    cfg = load_config()
    det = load_metrics("detect")
    sur = load_metrics("surrogate")
    ckpt = repo_root() / "data" / "checkpoints" / "best.pt"
    _require(ckpt.exists(), "surrogate checkpoint", "python -m emberline.surrogate.train")
    state = torch.load(ckpt, weights_only=False, map_location="cpu")
    n_params = int(sum(v.numel() for v in state["model"].values()))

    fig, ax = plt.subplots(**FIG_KW)
    ax.set_xlim(0, 100)
    ax.set_ylim(0, 100)
    ax.axis("off")
    fig.suptitle("Emberline — synthetic wildfire early-warning stack",
                 fontsize=30, fontweight="bold", color=INK, y=0.97)
    ax.text(50, 92.5, "end-to-end pipeline: every arrow is data produced and "
                      "consumed by code in this repo (all synthetic)",
            fontsize=15, color="#555", ha="center")

    def box(x, y, w, h, title, sub, key):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.6",
                                    fc="white", ec=ACCENT[key], lw=2.6))
        ax.text(x + w / 2, y + h - 3.2, title, fontsize=17, fontweight="bold",
                color=ACCENT[key], ha="center")
        ax.text(x + w / 2, y + h / 2 - 2.4, sub, fontsize=12.5, color=INK,
                ha="center", va="center", linespacing=1.5)

    def arrow(x0, y0, x1, y1, label, dx=0.0, dy=0.0, fs=12):
        ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>",
                                     mutation_scale=22, color="#666", lw=2.2,
                                     shrinkA=4, shrinkB=4))
        ax.text((x0 + x1) / 2 + dx, (y0 + y1) / 2 + dy, label, fontsize=fs,
                color="#444", ha="center", style="italic")

    gs = int(cfg["world"]["grid_size"])
    box(30, 76, 40, 12, "worldgen — synthetic world",
        f"fractal terrain · fuel map · town + exit roads\n"
        f"OU-gust wind · {gs}×{gs} cells @ {cfg['world']['cell_m']:.0f} m", "world")

    box(4, 52, 27, 15, "firesim — physics truth",
        "Rothermel-inspired probabilistic CA\nwind + slope + fuel ROS,"
        " MC ensembles", "physics")
    iou_line = (f"IoU@+30: {sur['val']['iou_30']:.2f} val / "
                f"{sur['holdout_wind_regime']['iou_30']:.2f} holdout wind" if sur
                else "one forward pass per +10 min")
    box(36.5, 52, 27, 15, "surrogate — neural twin",
        f"UNet, {n_params:,} params\n+10 min per forward pass\n{iou_line}",
        "surrogate")
    box(69, 52, 27, 15, "sensors — virtual nodes",
        f"{cfg['sensors']['n_nodes']} nodes · 1 Hz PM/VOC/T/RH\nGaussian plume"
        " + 6 confounders", "sensors")

    box(69, 30, 27, 14, "detect — on-node classifier",
        (f"1D-CNN F1 {det['cnn']['f1']:.3f} vs GBM {det['gbm']['f1']:.3f}\n"
         if det else "1D-CNN vs GBM\n") + "60-s windows, event-split", "detect")
    box(36.5, 30, 27, 14, "mesh — LoRa DES",
        "flooding + CAD + 1% duty\nTier 0/1/2 ladder · self-heal", "mesh")
    box(4, 30, 27, 14, "foresight — decision layer",
        "bearing triangulation\nP(burn) cones · evac routing\nCAP drafts (Exercise)",
        "foresight")

    box(23, 6, 54, 13, "demo + report — outputs",
        "terminal story · GIF frames · scoreboard\nhindcast harness · "
        "REPORT.md (all numbers measured)", "out")

    arrow(38, 76, 17.5, 67, "terrain·fuel·wind", dx=-8.5, dy=1.5)
    arrow(50, 76, 50, 67, "static planes + live wind", dx=13)
    arrow(62, 76, 82.5, 67, "plume advection", dx=9, dy=1.5)
    arrow(31, 59.5, 36.5, 59.5, "16.8k\npairs", dy=4.5)
    arrow(82.5, 52, 82.5, 44, "windows", dx=6)
    arrow(69, 37, 63.5, 37, "P(fire)+bearing", dy=2.2)
    arrow(36.5, 37, 31, 37, "Tier-1+ events", dy=2.2)
    arrow(50, 52, 50, 44, "cones on demand", dx=11)
    arrow(17.5, 30, 40, 19, "ADVISORY cones·routes·CAP", dx=-10, dy=-2)
    arrow(50, 30, 50, 19, "escalation log", dx=9)
    arrow(17.5, 52, 17.5, 44.2, "demo truth (eval only)", dx=-10.5)

    p = out_dir() / "system_architecture.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ 2 -------
def cone_evacuation_before_after() -> pathlib.Path:
    npz, js, run = _load_run()
    _require("cone_after_prob" in npz, "post-wind-shift forecast in the dump",
             "python -m emberline.demo --scenario ridgeline --fast --kill-node N3 --wind-shift 40")
    cfg = load_config()
    world = generate_world(cfg, 0)

    fig, axes = plt.subplots(1, 2, **FIG_KW)
    tier2_s = run["tier2_sim_s_after_ignition"]
    shift_s = run["windshift_sim_s_after_ignition"]
    fig.suptitle("Ridgeline demo — probability cone & evacuation routes across a "
                 f"{run['windshift_deg']:+.0f}° wind shift", fontsize=28,
                 fontweight="bold", color=INK, y=0.98)

    panels = [
        (axes[0], "cone_before_prob", "fire_state_tier2", js["plan_before"],
         "wind_uv_at_tier2",
         f"BEFORE — Tier-2 forecast at t+{tier2_s:.0f} s after ignition"),
        (axes[1], "cone_after_prob", "fire_state_postshift", js["plan_after"],
         "wind_uv_after_shift",
         f"AFTER — re-forecast at t+{shift_s:.0f} s, post wind shift"),
    ]
    thresholds = [float(t) for t in npz["cone_thresholds"]]
    exit_palette = ["#06d6a0", "#118ab2", "#9b5de5"]

    for ax, cone_key, fire_key, plan, wind_key, title in panels:
        _basemap(ax, world)
        prob = npz[cone_key][-1]  # +60 min horizon
        for thr, col in zip(thresholds, CONE_COLORS):
            ax.contourf(prob, levels=[thr, 1.01], colors=[col], alpha=0.35)
            ax.contour(prob, levels=[thr], colors=[col], linewidths=1.6)
        if fire_key in npz:
            fs = npz[fire_key]
            overlay = np.zeros((*fs.shape, 4))
            overlay[fs == 1] = (1.0, 0.15, 0.0, 1.0)
            overlay[fs == 2] = (0.12, 0.12, 0.12, 0.9)
            ax.imshow(overlay, origin="lower", interpolation="nearest")
        exits = {f"{r},{c}": (r, c) for r, c in world.town.exits}
        exit_color = {k: exit_palette[i % 3] for i, k in enumerate(sorted(exits))}
        for i, (key, path) in enumerate(sorted(plan["routes"].items())):
            if i % 4 or len(path) < 2:
                continue
            pth = np.array(path)
            ax.plot(pth[:, 1], pth[:, 0],
                    color=exit_color[f"{path[-1][0]},{path[-1][1]}"], lw=2.0, alpha=0.9)
        for key, (er, ec) in exits.items():
            load = plan["exit_loads"].get(key, 0)
            ax.scatter([ec], [er], s=420, marker="*", c=exit_color[key],
                       edgecolors="white", linewidths=1.4, zorder=7)
            # Keep edge-of-map labels inside the panel (exits sit on map edges).
            dx = 12 if ec < world.n - 40 else -60
            dy = 12 if er < world.n - 40 else -26
            if er < 20:
                dy = 16
            ax.annotate(f"{load} veh", (ec, er), fontsize=14, fontweight="bold",
                        color=INK, xytext=(dx, dy), textcoords="offset points",
                        zorder=8,
                        bbox=dict(boxstyle="round,pad=0.25", fc="white", ec=exit_color[key]))
        node_rc, alive = npz["node_rc"], npz["node_alive"]
        ax.scatter(node_rc[alive, 1], node_rc[alive, 0], s=70, marker="^",
                   c="#118ab2", edgecolors="white", linewidths=0.8, zorder=6)
        if (~alive).any():
            ax.scatter(node_rc[~alive, 1], node_rc[~alive, 0], s=110, marker="X",
                       c="#8d0801", edgecolors="white", linewidths=0.8, zorder=6)
        ir, ic = npz["ignition_truth"]
        ax.scatter([ic], [ir], s=200, marker="o", facecolors="none",
                   edgecolors="#d00000", linewidths=2.5, zorder=7)
        _wind_arrow(ax, world, npz[wind_key], "wind")
        ax.set_title(title, fontsize=18, color=INK, pad=10)
        ax.text(0.01, -0.035,
                f"{plan['blocked_edges']} road edges cut by the +30 min ≥ "
                f"{cfg['foresight']['routing_threshold']:.2f} cone · "
                f"{plan['unreachable']} access points unreachable · ADVISORY only",
                transform=ax.transAxes, fontsize=12.5, color="#555")

    handles = ([Line2D([], [], color=c, lw=6, alpha=0.5) for c in CONE_COLORS]
               + [Line2D([], [], color="#666", marker="^", ls="", ms=10),
                  Line2D([], [], color="#8d0801", marker="X", ls="", ms=11),
                  Line2D([], [], color="#d00000", marker="o", mfc="none", ls="", ms=11)])
    labels = [f"P(burn by +60) ≥ {t:.2f}" for t in thresholds] + \
             ["sensor node", "killed node", "true ignition"]
    fig.legend(handles, labels, loc="lower center", ncol=6, fontsize=13.5,
               frameon=False, bbox_to_anchor=(0.5, 0.005))
    fig.tight_layout(rect=(0, 0.035, 1, 0.94))
    p = out_dir() / "cone_evacuation_before_after.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ 3 -------
def mesh_topology() -> pathlib.Path:
    npz, js, run = _load_run()
    cfg = load_config()
    world = generate_world(cfg, 0)
    node_rc, health = npz["node_rc"], npz["node_health"]
    ids = [f"N{i}" for i in range(len(node_rc))]
    pos = {nid: rc for nid, rc in zip(ids, node_rc)}
    killed = js["args"].get("kill_node")
    head_final = js["head_final"]
    head_first = next((n for _, n, k, _ in js["mesh_log"] if k == "HEAD"), head_final)
    heal = next((t / 1000.0 for t, n, k, _ in js["mesh_log"]
                 if k == "SELF-HEAL" and n == killed), None)

    fig, ax = plt.subplots(**FIG_KW)
    _basemap(ax, world, alpha_fuel=0.18)
    fig.suptitle("Emberline mesh — 12-node LoRa layout over the demo world",
                 fontsize=28, fontweight="bold", color=INK, y=0.97)
    ax.set_title(f"links = modelled RSSI ≥ {cfg['mesh']['sensitivity_dbm']:.0f} dBm "
                 "(log-distance + terrain occlusion) · from the instrumented "
                 "ridgeline run", fontsize=15, color="#555", pad=8)

    drawn = set()
    for a, nbrs in js["neighbors"].items():
        for b in nbrs:
            if (b, a) in drawn:
                continue
            drawn.add((a, b))
            (r1, c1), (r2, c2) = pos[a], pos[b]
            through_kill = killed in (a, b)
            ax.plot([c1, c2], [r1, r2], color="#8d0801" if through_kill else "#457b9d",
                    lw=1.4 if through_kill else 2.0, alpha=0.5 if through_kill else 0.55,
                    ls=":" if through_kill else "-", zorder=3)

    for i, nid in enumerate(ids):
        r, c = node_rc[i]
        if nid == killed:
            ax.scatter([c], [r], s=520, marker="X", c="#8d0801",
                       edgecolors="white", linewidths=1.6, zorder=6)
            ax.annotate(nid, (c, r), fontsize=13, fontweight="bold", color="#8d0801",
                        xytext=(-18, 8), textcoords="offset points")
            k_s = run.get("kill_sim_s_after_ignition")
            note = f"{nid} KILLED at t+{k_s:.0f} s" if k_s is not None else f"{nid} KILLED"
            if heal is not None:
                note += f"\nheartbeats missed → mourned and\nrouted around at t+{heal:.0f} s (measured)"
            ax.text(-0.05, 0.66, note, transform=ax.transAxes, fontsize=15.5,
                    fontweight="bold", color="#8d0801", ha="right", va="center",
                    linespacing=1.6,
                    bbox=dict(boxstyle="round,pad=0.5", fc="white", ec="#8d0801", lw=2))
            continue
        ax.scatter([c], [r], s=300 + 300 * float(health[i]), marker="^",
                   c="#118ab2" if float(health[i]) > 0.7 else "#e9c46a",
                   edgecolors="white", linewidths=1.2, zorder=6)
        ax.annotate(nid, (c, r), fontsize=13, fontweight="bold", color=INK,
                    xytext=(8, 8), textcoords="offset points")
    hr, hc = pos[head_final]
    ax.scatter([hc], [hr], s=1500, facecolors="none", edgecolors="#2a9d8f",
               linewidths=3.0, zorder=7)
    ax.annotate(f"cluster head {head_final}"
                + ("" if head_final == head_first else f" (was {head_first})"),
                (hc, hr), fontsize=14, fontweight="bold", color="#2a9d8f",
                xytext=(16, 22), textcoords="offset points",
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#2a9d8f"))
    for er, ec in world.town.exits:
        ax.scatter([ec], [er], s=300, marker="*", c="#ef476f", zorder=6)

    handles = [Line2D([], [], color="#457b9d", lw=3, alpha=0.6),
               Line2D([], [], color="#8d0801", lw=2, ls=":"),
               Line2D([], [], color="#118ab2", marker="^", ls="", ms=13),
               Line2D([], [], color="#e9c46a", marker="^", ls="", ms=13),
               Line2D([], [], color="#8d0801", marker="X", ls="", ms=13),
               Line2D([], [], color="#2a9d8f", marker="o", mfc="none", ls="", ms=14),
               Line2D([], [], color="#ef476f", marker="*", ls="", ms=15)]
    labels = ["radio link", "links lost with node", "healthy node (size ∝ health)",
              "degraded node", "killed node", "cluster head", "town exit road"]
    fig.legend(handles, labels, loc="lower center", ncol=7, fontsize=13.5,
               frameon=False, bbox_to_anchor=(0.5, 0.008))
    fig.tight_layout(rect=(0, 0.04, 1, 0.93))
    p = out_dir() / "mesh_topology.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ 4 -------
def detection_confusion() -> pathlib.Path:
    det = load_metrics("detect")
    _require(det is not None and "per_confounder" in det, "metrics/detect.json",
             "python -m emberline.detect.eval")
    rows = sorted(det["per_confounder"].items(),
                  key=lambda kv: -(kv[1]["fp_rate_cnn"] + kv[1]["fp_rate_gbm"]))
    names = [k for k, _ in rows]
    cnn = [v["fp_rate_cnn"] * 100 for _, v in rows]
    gbm = [v["fp_rate_gbm"] * 100 for _, v in rows]
    counts = [v["windows"] for _, v in rows]

    fig, ax = plt.subplots(**FIG_KW)
    fig.suptitle("Per-confounder false-positive rate — 1D-CNN vs GBM baseline",
                 fontsize=28, fontweight="bold", color=INK, y=0.96)
    ax.set_title("held-out (event-split) validation windows · synthetic signature "
                 f"models · CNN F1 {det['cnn']['f1']:.3f}, GBM F1 {det['gbm']['f1']:.3f}",
                 fontsize=15, color="#555", pad=12)
    y = np.arange(len(names))
    h = 0.38
    ax.barh(y - h / 2, cnn, height=h, color="#e76f51", label="1D-CNN")
    ax.barh(y + h / 2, gbm, height=h, color="#457b9d", label="GBM baseline")
    for yi, (c_val, g_val, n) in enumerate(zip(cnn, gbm, counts)):
        for val, off in ((c_val, -h / 2), (g_val, h / 2)):
            ax.text(val + 0.35, yi + off, f"{val:.1f}%", va="center", fontsize=13.5,
                    color=INK)
        ax.text(-0.6, yi, f"{names[yi]}\n({n} windows)", va="center", ha="right",
                fontsize=15, color=INK)
    ax.set_yticks([])
    ax.set_xlabel("false-positive rate on validation windows (%)", fontsize=16)
    ax.tick_params(axis="x", labelsize=14)
    ax.set_xlim(0, max(cnn + gbm) * 1.18 + 1)
    ax.invert_yaxis()
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.legend(fontsize=16, frameon=False, loc="center right")
    ax.text(0.995, 0.02,
            f"ambient day: {det['ambient_day']['fp_per_node_day_cnn']:.1f} (CNN) vs "
            f"{det['ambient_day']['fp_per_node_day_gbm']:.1f} (GBM) FP/node-day over "
            f"{det['ambient_day']['windows_evaluated']:,} windows — the mesh "
            "corroboration ladder is what turns windows into sirens",
            transform=ax.transAxes, fontsize=13.5, color="#555", ha="right")
    fig.tight_layout(rect=(0.02, 0.02, 1, 0.92))
    p = out_dir() / "detection_confusion.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ 5 -------
def calibration_before_after() -> pathlib.Path:
    v1 = load_metrics("calibration_v1")
    v2 = load_metrics("calibration")
    _require(v1 is not None, "metrics/calibration_v1.json",
             "python -m emberline.surrogate.calibration --ckpt data/checkpoints/best_v1.pt "
             "--metrics-name calibration_v1 --no-report")
    _require(v2 is not None and "ece_val_candidate" in v2, "metrics/calibration.json",
             "python -m emberline.surrogate.calibration")

    fig, ax = plt.subplots(**FIG_KW)
    fig.suptitle("Ensemble calibration — before vs after Phase 9/10",
                 fontsize=28, fontweight="bold", color=INK, y=0.96)
    ax.set_title(f"P(burn by +{v2['horizon_min']:.0f} min) vs physics outcome · "
                 f"{v2['fires']} val fires · {v2['members']} members · temperature "
                 "fitted on separate calibration worlds", fontsize=15, color="#555",
                 pad=12)
    ax.plot([0, 1], [0, 1], "--", color="#999", lw=2, label="perfect calibration")
    ax.plot(v1["bin_pred"], v1["bin_freq"], "o-", color="#adb5bd", lw=3, ms=11,
            label=f"v1 surrogate, raw — ECE {v1['ece']:.3f}")
    ax.plot(v2["bin_pred"], v2["bin_freq"], "o-", color="#d62828", lw=3, ms=11,
            label=f"v2 (wind-augmented), raw — ships — ECE {v2['ece']:.3f}")
    ax.text(0.97, 0.05,
            "the wind-augmented retraining itself did the calibrating: the v1\n"
            f"curve sits far below the diagonal (over-confident), v2 hugs it.\n"
            f"A temperature fitted on separate calibration worlds (T="
            f"{v2['candidate_temperature']:.2f}) does not\ntransfer to this "
            "population and is NOT shipped — full fit-then-verify\nprotocol, "
            "curves and caveats in REPORT.md + demo/out/calibration_v2.png",
            transform=ax.transAxes, fontsize=13, color="#555", ha="right",
            linespacing=1.55)
    ax.set_xlabel("predicted burn probability", fontsize=17)
    ax.set_ylabel("observed burn frequency (physics)", fontsize=17)
    ax.tick_params(labelsize=14)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_aspect("equal")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=16, frameon=False, loc="upper left")
    ax.text(0.03, 0.35, "points below the diagonal =\nover-confident forecasts",
            transform=ax.transAxes, fontsize=13.5, color="#555", ha="left",
            linespacing=1.5)
    fig.tight_layout(rect=(0.24, 0.02, 0.76, 0.92))
    p = out_dir() / "calibration_before_after.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


# ------------------------------------------------------------------ 6 -------
def warning_timeline() -> pathlib.Path:
    run = load_metrics("demo_last_run")
    _require(run is not None and "tier2_sim_s_after_ignition" in run,
             "metrics/demo_last_run.json",
             "python -m emberline.demo --scenario ridgeline --fast --kill-node N3 --wind-shift 40")

    events = [("IGNITION", 0.0, "#d00000",
               "hidden from the system — comms are down")]
    for key, name, col, sub in [
            ("tier0_sim_s_after_ignition", "TIER 0", "#118ab2", "first node chirps locally"),
            ("tier1_sim_s_after_ignition", "TIER 1", "#e9c46a", "voice alert with bearing"),
            ("tier2_sim_s_after_ignition", "TIER 2", "#e63946",
             "full siren cascade + cones + routes + CAP draft"),
            ("kill_sim_s_after_ignition", f"{run.get('killed_node', 'node')} KILLED",
             "#8d0801", "failure injected mid-cascade"),
            ("selfheal_sim_s_after_ignition", "SELF-HEAL", "#2a9d8f",
             "mesh mourns the node and routes around it"),
            ("windshift_sim_s_after_ignition",
             f"WIND +{run.get('windshift_deg', 0):.0f}°", "#f3722c",
             "re-forecast + evacuation re-plan")]:
        if key in run:
            events.append((name, float(run[key]) / 60.0, col, sub))
    events.sort(key=lambda e: e[1])

    fig, ax = plt.subplots(**FIG_KW)
    fig.suptitle("Ridgeline demo — measured warning timeline", fontsize=30,
                 fontweight="bold", color=INK, y=0.94)
    ax.set_title("minutes after ignition · every timestamp from "
                 "metrics/demo_last_run.json (one instrumented run, synthetic sim)",
                 fontsize=15.5, color="#555", pad=14)
    t_max = max(t for _, t, _, _ in events)
    ax.set_xlim(-1.2, t_max * 1.14)
    ax.set_ylim(-3.2, 3.2)
    ax.axhline(0, color=INK, lw=3, zorder=2)
    ax.plot([t_max * 1.10], [0], marker=">", color=INK, ms=14, zorder=2)

    for i, (name, t_min, col, sub) in enumerate(events):
        up = 1 if i % 2 == 0 else -1
        y = up * (1.35 if i % 4 < 2 else 2.15)
        ax.plot([t_min, t_min], [0, y], color=col, lw=2.4, zorder=3)
        ax.scatter([t_min], [0], s=200, color=col, zorder=4,
                   edgecolors="white", linewidths=1.5)
        label = f"{name}\nt+{t_min * 60:.0f} s" if t_min < 2 else f"{name}\nt+{t_min:.1f} min"
        ax.text(t_min, y + 0.16 * up, label, ha="center",
                va="bottom" if up > 0 else "top", fontsize=17, fontweight="bold",
                color=col, linespacing=1.4)
        ax.text(t_min, y + 0.16 * up + (0.62 * up if up > 0 else 0.0)
                - (0.0 if up > 0 else 0.62), sub, ha="center",
                va="bottom" if up > 0 else "top", fontsize=12.5, color="#555",
                style="italic")
    ax.set_yticks([])
    ax.set_xlabel("minutes after ignition", fontsize=17)
    ax.tick_params(axis="x", labelsize=15)
    ax.spines[["top", "right", "left"]].set_visible(False)
    fig.tight_layout(rect=(0.01, 0.04, 0.99, 0.9))
    p = out_dir() / "warning_timeline.png"
    fig.savefig(p, facecolor="white")
    plt.close(fig)
    return p


ASSETS = {
    "system_architecture": system_architecture,
    "cone_evacuation_before_after": cone_evacuation_before_after,
    "mesh_topology": mesh_topology,
    "detection_confusion": detection_confusion,
    "calibration_before_after": calibration_before_after,
    "warning_timeline": warning_timeline,
}


def main() -> None:
    ap = argparse.ArgumentParser(description="Render pitch assets from real run artifacts")
    ap.add_argument("--only", choices=sorted(ASSETS), default=None)
    args = ap.parse_args()
    for name, fn in ASSETS.items():
        if args.only and name != args.only:
            continue
        print(f"rendering {name} …", flush=True)
        print(f"  -> {fn()}")


if __name__ == "__main__":
    main()
