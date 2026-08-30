"""CLI: ``python -m emberline.firesim --animate`` saves a spread GIF.

Also ``--ensemble`` renders burn-probability maps from a Monte Carlo run.
"""

from __future__ import annotations

import argparse
import pathlib
import time

import imageio.v2 as imageio
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from ..config import load_config, repo_root, rng_for
from ..viz import FUEL_CMAP, hillshade
from ..worldgen import generate_world
from . import BURNED, BURNING, FireSim
from .ensemble import run_ensemble


def render_frame(world, state, t_min: float) -> np.ndarray:
    fig, ax = plt.subplots(figsize=(5, 5), dpi=100)
    ax.imshow(world.fuel, cmap=FUEL_CMAP, vmin=0, vmax=4, origin="lower", alpha=0.6,
              interpolation="nearest")
    overlay = np.zeros((*state.shape, 4))
    overlay[state == BURNING] = (1.0, 0.25, 0.0, 1.0)
    overlay[state == BURNED] = (0.15, 0.15, 0.15, 0.9)
    ax.imshow(overlay, origin="lower", interpolation="nearest")
    ax.set_title(f"t = {t_min:.0f} min")
    ax.axis("off")
    fig.tight_layout(pad=0.2)
    fig.canvas.draw()
    buf = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return buf


def main() -> None:
    ap = argparse.ArgumentParser(description="Emberline physics fire simulator")
    ap.add_argument("--animate", action="store_true")
    ap.add_argument("--ensemble", action="store_true")
    ap.add_argument("--minutes", type=float, default=90.0)
    ap.add_argument("--world-id", type=int, default=0)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = load_config()
    world = generate_world(cfg, args.world_id)
    out = pathlib.Path(args.out) if args.out else repo_root() / "demo" / "out" / "firesim"
    out.mkdir(parents=True, exist_ok=True)

    # Ignite upwind of town so the fire threatens it.
    r0, c0, r1, c1 = world.town.district
    _, wdir = world.wind.at(0.0)
    ig_r = int(np.clip((r0 + r1) / 2 - 70 * np.sin(wdir), 5, world.n - 6))
    ig_c = int(np.clip((c0 + c1) / 2 - 70 * np.cos(wdir), 5, world.n - 6))

    if args.animate or not args.ensemble:
        sim = FireSim(world, rng_for(cfg, "firesim-demo", args.world_id))
        sim.ignite(ig_r, ig_c)
        frames, t0 = [], time.perf_counter()
        step_min = 5.0
        for k in range(int(args.minutes / step_min) + 1):
            if args.animate:
                frames.append(render_frame(world, sim.state, k * step_min))
            sim.run(step_min)
        wall = time.perf_counter() - t0
        print(f"simulated {args.minutes:.0f} sim-min in {wall:.2f} s "
              f"({int(sim.burned_or_burning.sum())} cells touched)")
        if args.animate:
            imageio.mimsave(out / "spread.gif", frames, fps=4)
            print(f"saved {out / 'spread.gif'}")

    if args.ensemble:
        e = cfg["firesim"]["ensemble"]
        t0 = time.perf_counter()
        res = run_ensemble(world, (ig_r, ig_c), [10, 30, 60], int(e["n_members"]),
                           rng_for(cfg, "firesim-ensemble", args.world_id))
        wall = time.perf_counter() - t0
        print(f"{res.n_members}-member ensemble x 60 min in {wall:.1f} s")
        fig, axes = plt.subplots(1, 3, figsize=(13, 4.5), dpi=100)
        for ax, h, pm in zip(axes, res.horizons_min, res.prob):
            ax.imshow(hillshade(world.elevation), cmap="gray", origin="lower", alpha=0.6)
            im = ax.imshow(pm, cmap="inferno", vmin=0, vmax=1, origin="lower", alpha=0.75)
            ax.set_title(f"P(burned) by +{h:.0f} min")
            ax.axis("off")
        fig.colorbar(im, ax=axes, shrink=0.75)
        fig.savefig(out / "ensemble_prob.png")
        print(f"saved {out / 'ensemble_prob.png'}")


if __name__ == "__main__":
    main()
