"""Demo frame rendering: truth fire + probability cone + town + routes."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..viz import FUEL_CMAP, hillshade

CONE_COLORS = ["#ffd166", "#f3722c", "#d62828"]  # low->high probability


def render_scene(
    world,
    fire_state: np.ndarray | None,
    cone_prob: np.ndarray | None,  # (n_thresh applied to this single map)
    thresholds: list[float],
    nodes,
    routes: dict | None,
    title: str,
    path: pathlib.Path,
) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(6.4, 6.4), dpi=100)
    ax.imshow(hillshade(world.elevation), cmap="gray", origin="lower", alpha=0.75)
    ax.imshow(world.fuel, cmap=FUEL_CMAP, vmin=0, vmax=4, origin="lower", alpha=0.30,
              interpolation="nearest")

    for a, b, data in world.town.roads.edges(data=True):
        cells = np.array(data["cells"])
        ax.plot(cells[:, 1], cells[:, 0], color="#333333",
                lw=1.8 if data["kind"] == "exit" else 0.6, alpha=0.8)

    if cone_prob is not None:
        for thr, col in zip(thresholds, CONE_COLORS):
            ax.contourf(cone_prob, levels=[thr, 1.01], colors=[col], alpha=0.35)
            ax.contour(cone_prob, levels=[thr], colors=[col], linewidths=1.2)

    if fire_state is not None:
        overlay = np.zeros((*fire_state.shape, 4))
        overlay[fire_state == 1] = (1.0, 0.2, 0.0, 1.0)
        overlay[fire_state == 2] = (0.1, 0.1, 0.1, 0.85)
        ax.imshow(overlay, origin="lower", interpolation="nearest")

    if routes:
        for path_nodes in list(routes.values())[::6]:  # thin out for legibility
            p = np.array(path_nodes)
            ax.plot(p[:, 1], p[:, 0], color="#06d6a0", lw=1.6, alpha=0.9)

    for nd in nodes:
        ax.scatter([nd.c], [nd.r], s=42, marker="^",
                   c="#118ab2" if nd.alive else "#7a7a7a",
                   edgecolors="white", linewidths=0.6, zorder=6)
        ax.annotate(nd.node_id, (nd.c, nd.r), fontsize=6, color="white",
                    xytext=(3, 3), textcoords="offset points")

    for er, ec in world.town.exits:
        ax.scatter([ec], [er], s=120, marker="*", c="#ef476f", zorder=6)

    ax.set_title(title, fontsize=10)
    ax.set_xlim(0, world.n)
    ax.set_ylim(0, world.n)
    ax.axis("off")
    fig.tight_layout(pad=0.3)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    fig.savefig(path)
    plt.close(fig)
    return buf
