"""Render the system and data-flow diagrams to PNG with matplotlib.

    python docs/diagrams/render_diagrams.py

The repository has no Mermaid renderer, so slides get these two PNGs drawn
directly. The Mermaid sources (``*.mmd`` in this folder) remain the
authoritative diagrams; this script mirrors their content (boxes, labels and
built/planned styling) and nothing else. Outputs:

    docs/diagrams/system.png       (matches system.mmd)
    docs/diagrams/data_flow.png    (matches data_flow.mmd)

No repository code is imported; this is documentation tooling only.
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

HERE = pathlib.Path(__file__).resolve().parent

BUILT_FILL, BUILT_EDGE = "#e8f1e4", "#2e5d34"
PLAN_FILL, PLAN_EDGE = "#fff4e5", "#b8860b"
INK = "#1b1b1b"
LINE = "#333333"
GROUP_EDGE = "#8a8a8a"


# ------------------------------------------------------------ primitives ----
def box(ax, xy, w, h, text, planned=False, fontsize=9.5):
    x, y = xy
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.06",
                                linewidth=1.6,
                                edgecolor=PLAN_EDGE if planned else BUILT_EDGE,
                                facecolor=PLAN_FILL if planned else BUILT_FILL,
                                linestyle=(0, (5, 3)) if planned else "solid"))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fontsize,
            color=INK, linespacing=1.35)
    return (x, y, w, h)


def _edge_points(a, b):
    ax1, ay1, aw, ah = a
    bx1, by1, bw, bh = b
    if bx1 >= ax1 + aw:  # b right of a
        return (ax1 + aw, ay1 + ah / 2), (bx1, by1 + bh / 2)
    if bx1 + bw <= ax1:  # b left of a
        return (ax1, ay1 + ah / 2), (bx1 + bw, by1 + bh / 2)
    if by1 >= ay1 + ah:  # b above a
        return (ax1 + aw / 2, ay1 + ah), (bx1 + bw / 2, by1)
    return (ax1 + aw / 2, ay1), (bx1 + bw / 2, by1 + bh)  # b below a


def arrow(ax, a, b, planned=False, label=None, label_dy=0.12):
    start, end = _edge_points(a, b)
    ax.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.4, color=PLAN_EDGE if planned else LINE,
                                 linestyle=(0, (4, 3)) if planned else "solid"))
    if label:
        mx, my = (start[0] + end[0]) / 2, (start[1] + end[1]) / 2
        ax.text(mx, my + label_dy, label, ha="center", va="bottom", fontsize=7.5,
                color="#444444",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))


def elbow_down(ax, x, y_from, y_to, planned=False):
    """Vertical connector with an arrowhead at the bottom."""
    ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to), arrowstyle="-|>", mutation_scale=14,
                                 linewidth=1.4, color=PLAN_EDGE if planned else LINE,
                                 linestyle=(0, (4, 3)) if planned else "solid"))


def group(ax, x, y, w, h, title):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                                linewidth=1.0, edgecolor=GROUP_EDGE, facecolor="none",
                                linestyle=(0, (2, 2))))
    ax.text(x + 0.1, y + h - 0.08, title, ha="left", va="top", fontsize=10.5,
            color="#333333", fontweight="bold")


def legend_row(ax, x, y):
    """Horizontal legend: [built] label   [planned] label."""
    ax.add_patch(FancyBboxPatch((x, y), 0.45, 0.28, boxstyle="round,pad=0.02",
                                fc=BUILT_FILL, ec=BUILT_EDGE, lw=1.4))
    ax.text(x + 0.55, y + 0.14, "built and tested in this repo (simulation)", va="center",
            fontsize=8.5, color=INK)
    x2 = x + 4.3
    ax.add_patch(FancyBboxPatch((x2, y), 0.45, 0.28, boxstyle="round,pad=0.02",
                                fc=PLAN_FILL, ec=PLAN_EDGE, lw=1.4, linestyle=(0, (5, 3))))
    ax.text(x2 + 0.55, y + 0.14, "PLANNED / NOT BUILT", va="center", fontsize=8.5, color=INK)


# ------------------------------------------------------------ system.png ----
def render_system(path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 7.2), dpi=120)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 7.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.2, 6.95, "Emberline system - sense → classify → mesh/escalate → Foresight → "
                       "human-approved alert", fontsize=14, fontweight="bold", color=INK, va="top")
    ax.text(0.2, 6.55, "Solid = built and measured as software simulation (synthetic data). "
                       "Dashed = PLANNED or NOT BUILT.", fontsize=9.5, color="#444444", va="top")

    group(ax, 0.2, 2.0, 3.1, 4.2, "1. Sense")
    group(ax, 3.6, 2.0, 2.9, 4.2, "2. Classify on node")
    group(ax, 6.8, 2.0, 3.1, 4.2, "3. Escalate over mesh")
    group(ax, 10.1, 0.3, 5.7, 5.9, "4. Foresight")

    hw = box(ax, (0.3, 4.4), 2.9, 1.3, "Physical node\nESP32-S3 + BME680 + PMS5003\n(NOT BUILT)",
             planned=True, fontsize=9)
    vs = box(ax, (0.3, 2.3), 2.9, 1.5, "Virtual sensors\nemberline/sensors\nplume + confounders + noise")
    det = box(ax, (3.8, 3.2), 2.5, 1.7, "60-s window classifier\nemberline/detect\n"
                                        "1D-CNN 9,129 params\nvs GBM baseline")
    radio = box(ax, (7.0, 4.4), 2.7, 1.3, "LoRa radios, solar, siren\n(NOT BUILT)", planned=True)
    des = box(ax, (7.0, 2.3), 2.7, 1.6, "Discrete-event mesh sim\nemberline/mesh\n"
                                        "Tier 0 chirp → Tier 1 voice\n→ Tier 2 cascade", fontsize=9)
    twin = box(ax, (10.3, 4.5), 2.5, 1.2, "Synthetic twin\nemberline/worldgen")
    real = box(ax, (13.1, 4.5), 2.5, 1.2, "Real terrain / fuel / roads\nemberline/adapters (STUBS)",
               planned=True, fontsize=8.5)
    sur = box(ax, (10.3, 2.7), 2.5, 1.3, "Neural surrogate ensemble\nemberline/surrogate\n"
                                         "cones +10/+30/+60", fontsize=9)
    route = box(ax, (13.1, 2.7), 2.5, 1.3, "Capacity-aware routing\nemberline/foresight/routing\n"
                                           "ADVISORY", fontsize=9)
    cap = box(ax, (13.1, 0.6), 2.5, 1.3, "CAP draft → outbox/\nemberline/foresight/cap\n"
                                         "status: Exercise", fontsize=9)
    human = box(ax, (10.3, 0.6), 2.5, 1.3, "Human approval\ncounty / fire district\n"
                                           "(integration NOT BUILT)", planned=True, fontsize=9)

    arrow(ax, hw, det, planned=True, label="1 Hz CSV (planned contract)")
    arrow(ax, vs, det)
    arrow(ax, det, des, label="P(fire), bearing")
    arrow(ax, radio, des, planned=True)
    arrow(ax, des, sur, label="Tier-1+ event")
    arrow(ax, twin, sur)
    arrow(ax, real, twin, planned=True)
    arrow(ax, sur, route)
    arrow(ax, route, cap)
    arrow(ax, cap, human)

    legend_row(ax, 0.4, 1.2)
    ax.text(15.8, 0.12, "docs/diagrams/system.png - mirrors system.mmd", ha="right",
            fontsize=7.5, color="#777777")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# --------------------------------------------------------- data_flow.png ----
def render_data_flow(path: pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(16, 9.2), dpi=120)
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.2)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.text(0.2, 9.0, "Emberline data flow - the synthetic / real boundary", fontsize=14,
            fontweight="bold", color=INK, va="top")
    ax.text(0.2, 8.6, "Left: every dataset that exists, all generated from seed 1337.  "
                      "Right: the planned real-data pipeline; none of it is built.",
            fontsize=9.5, color="#444444", va="top")
    legend_row(ax, 0.3, 7.95)

    group(ax, 0.2, 0.3, 10.2, 7.45, "SYNTHETIC - built and measured (this repo)")
    group(ax, 10.9, 0.3, 4.9, 7.45, "REAL - PLANNED, NOT BUILT")

    # Sources row.
    ys, hs = 6.2, 1.0
    seed = box(ax, (0.4, ys), 2.1, hs, "config.yaml seed 1337\nrng_for(stream, world_id)", fontsize=8.5)
    wg = box(ax, (2.8, ys), 4.3, hs,
             "worldgen\nworld 0 demo/hindcast · 1-160 train/val/holdout\n"
             "300-309 + 320-324 detect layouts · 161-172 calibration", fontsize=7.8)
    fs = box(ax, (7.4, ys), 2.6, hs, "firesim (physics truth)\nRothermel-style CA", fontsize=8.5)
    arrow(ax, seed, wg)
    arrow(ax, wg, fs)

    # Distribution bus: worlds + physics truth feed the three pipelines.
    bus_y = 5.75
    cols_x = [0.4, 3.7, 7.0]
    cw = 3.0
    ax.plot([wg[0] + wg[2] / 2, wg[0] + wg[2] / 2], [ys, bus_y], color=LINE, lw=1.4)
    ax.plot([fs[0] + fs[2] / 2, fs[0] + fs[2] / 2], [ys, bus_y], color=LINE, lw=1.4)
    ax.plot([cols_x[0] + cw / 2, cols_x[2] + cw / 2], [bus_y, bus_y], color=LINE, lw=1.4)
    ax.text(cols_x[1] + cw / 2, bus_y + 0.06, "worlds + physics truth", ha="center", va="bottom",
            fontsize=7.5, color="#444444",
            bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.9))

    rows_y = [4.55, 3.35, 2.15, 0.85]
    rh = 0.95
    half = (cw - 0.2) / 2
    # Columns 1 and 2 are fed at their centre; column 3's feed lands on the
    # hindcast half-box (right half of its top row).
    for cx in (cols_x[0], cols_x[1]):
        elbow_down(ax, cx + cw / 2, bus_y, rows_y[0] + rh)
    elbow_down(ax, cols_x[2] + cw - half / 2, bus_y, rows_y[0] + rh)

    # Column 1: surrogate.
    c1 = [box(ax, (cols_x[0], rows_y[0]), cw, rh, "data/surrogate/world_XXXX.npz\n160 shards · 16,841 (t, t+10) pairs", fontsize=8),
          box(ax, (cols_x[0], rows_y[1]), cw, rh, "split BY WORLD\ntrain · val 20 % · held-out wind 80-130°", fontsize=8),
          box(ax, (cols_x[0], rows_y[2]), cw, rh, "data/checkpoints/best.pt (v2)\nbest_v1.pt (archived)", fontsize=8),
          box(ax, (cols_x[0], rows_y[3]), cw, rh, "metrics/surrogate.json\nmetrics/calibration.json", fontsize=8)]
    # Column 2: detection.
    c2 = [box(ax, (cols_x[1], rows_y[0]), cw, rh, "sensors: plume + baselines\n+ 6 authored confounders + noise", fontsize=8),
          box(ax, (cols_x[1], rows_y[1]), cw, rh, "data/detect/windows.npz\n(N, 4, 60) windows + event_id", fontsize=8),
          box(ax, (cols_x[1], rows_y[2]), cw, rh, "split BY EVENT\n25 % of events → val", fontsize=8),
          box(ax, (cols_x[1], rows_y[3]), cw, rh, "detect_cnn.pt · detect_gbm.joblib\n→ metrics/detect.json", fontsize=8)]
    # Column 3: hindcast (top, scenarios beside it) and the demo below; both
    # consume the checkpoints of columns 1 and 2 (stated in the box text).
    x3 = cols_x[2]
    scen = box(ax, (x3, rows_y[0]), half, rh, "scenarios/*.yaml\n6 hand-authored\n(not fire records)", fontsize=7.4)
    hind = box(ax, (x3 + cw - half, rows_y[0]), half, rh, "foresight.hindcast\nworld 0 + physics\n+ CNN + mesh", fontsize=7.4)
    hmet = box(ax, (x3, rows_y[1]), cw, rh, "metrics/hindcast.json\n6 rows: tier minutes, warning minutes gained", fontsize=7.8)
    demo = box(ax, (x3, rows_y[2]), cw, rh, "demo (canonical run)\nworld 0 + physics + best.pt + detect_cnn.pt", fontsize=7.8)
    dmet = box(ax, (x3, rows_y[3]), cw, rh, "metrics/demo_last_run.json\ndemo/out/last_run_artifacts.* · pitch PNGs", fontsize=7.8)
    for col in (c1, c2):
        for a, b in zip(col, col[1:]):
            arrow(ax, a, b)
    arrow(ax, scen, hind)            # scenarios feed the hindcast
    arrow(ax, hind, hmet)            # hindcast -> its metrics
    arrow(ax, demo, dmet)            # demo -> metrics/artifacts

    # Real pipeline.
    rx, rw = 11.1, 4.5
    node = box(ax, (rx, 6.05), rw, 1.15,
               "Physical node  ESP32-S3 + BME680/688 + PMS5003\n1 Hz CSV:\nms,pm1,pm25,pm10,gas_ohms,temp_c,rh,press_hpa",
               planned=True, fontsize=7.8)
    sess = box(ax, (rx, 4.75), rw, 0.95,
               "Event-labelled sessions: ambient · bbq · wood_stove\nvehicle · fog · aerosol · supervised-burn smoke",
               planned=True, fontsize=8)
    rfeat = box(ax, (rx, 3.45), rw, 0.95,
                "Real-feature contract: PM2.5 mean/max/slope\nlog(gas / rolling 10-min baseline) · RH mean · temp mean",
                planned=True, fontsize=8)
    rgbm = box(ax, (rx, 2.3), rw, 0.75, "GBM retrained on real sessions (session-level split)",
               planned=True, fontsize=8)
    rrep = box(ax, (rx, 1.35), rw, 0.6, "Separate real-data report - never merged with synthetic tables",
               planned=True, fontsize=7.6)
    adapt = box(ax, (rx, 0.45), 2.05, 0.6, "adapters (stubs)\nUSGS · LANDFIRE · OSM", planned=True, fontsize=7.5)
    rhc = box(ax, (rx + 2.45, 0.45), 2.05, 0.6, "real hindcasts on historical fires\n(same YAML format as scenarios/)",
              planned=True, fontsize=7)
    arrow(ax, node, sess, planned=True)
    arrow(ax, sess, rfeat, planned=True)
    arrow(ax, rfeat, rgbm, planned=True)
    arrow(ax, rgbm, rrep, planned=True)
    arrow(ax, adapt, rhc, planned=True)

    # Boundary.
    ax.plot([10.65, 10.65], [0.3, 7.75], color="#b8860b", lw=2.0, linestyle=(0, (6, 4)))
    ax.text(10.65, 7.85, "no real byte has crossed this line", ha="center", va="bottom",
            fontsize=8.5, color="#b8860b", fontweight="bold")

    ax.text(15.8, 0.12, "docs/diagrams/data_flow.png - mirrors data_flow.mmd", ha="right",
            fontsize=7.5, color="#777777")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main() -> None:
    render_system(HERE / "system.png")
    render_data_flow(HERE / "data_flow.png")
    print(f"wrote {HERE / 'system.png'} and {HERE / 'data_flow.png'}")


if __name__ == "__main__":
    main()
