"""Shared matplotlib helpers (headless): fuel colormap, hillshade, world plots."""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LightSource, ListedColormap

# water, grass, brush, timber, urban
FUEL_COLORS = ["#3d6fb4", "#c9d97a", "#8fae5a", "#2e5d34", "#9a9a9a"]
FUEL_CMAP = ListedColormap(FUEL_COLORS)


def hillshade(elevation: np.ndarray) -> np.ndarray:
    ls = LightSource(azdeg=315, altdeg=45)
    return ls.hillshade(elevation, vert_exag=2.0)


def save_terrain_png(world, path: str | pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
    ax.imshow(hillshade(world.elevation), cmap="gray", origin="lower")
    im = ax.imshow(world.elevation, cmap="terrain", alpha=0.55, origin="lower")
    fig.colorbar(im, ax=ax, shrink=0.8, label="elevation (m)")
    ax.set_title(f"Terrain (world {world.world_id})")
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_fuel_png(world, path: str | pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
    ax.imshow(world.fuel, cmap=FUEL_CMAP, vmin=0, vmax=4, origin="lower",
              interpolation="nearest")
    ax.set_title("Fuel: water/grass/brush/timber/urban")
    handles = [plt.Rectangle((0, 0), 1, 1, fc=c) for c in FUEL_COLORS]
    ax.legend(handles, ["water", "grass", "brush", "timber", "urban"],
              loc="upper right", fontsize=7)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def save_town_png(world, path: str | pathlib.Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 6), dpi=110)
    ax.imshow(hillshade(world.elevation), cmap="gray", origin="lower", alpha=0.7)
    town = world.town
    for a, b, data in town.roads.edges(data=True):
        cells = np.array(data["cells"])
        color = "#d94f00" if data["kind"] == "exit" else "#444444"
        ax.plot(cells[:, 1], cells[:, 0], color=color,
                lw=2.0 if data["kind"] == "exit" else 0.8)
    bm = town.building_mask
    rr, cc = np.where(bm)
    ax.scatter(cc, rr, s=3, c="#7a2d2d", marker="s")
    for er, ec in [(e[0], e[1]) for e in town.exits]:
        ax.scatter([ec], [er], s=90, c="#d94f00", marker="*", zorder=5)
    r0, c0, r1, c1 = town.district
    ax.set_title(f"Town: {len(town.buildings)} buildings, {len(town.exits)} exits (stars)")
    ax.set_xlim(0, world.n)
    ax.set_ylim(0, world.n)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)
