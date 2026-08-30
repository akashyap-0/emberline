"""CLI: ``python -m emberline.worldgen --preview`` saves world PNGs."""

from __future__ import annotations

import argparse
import pathlib

from ..config import load_config, repo_root
from ..viz import save_fuel_png, save_terrain_png, save_town_png
from . import generate_world


def main() -> None:
    ap = argparse.ArgumentParser(description="Emberline world generator")
    ap.add_argument("--preview", action="store_true", help="save terrain/fuel/town PNGs")
    ap.add_argument("--world-id", type=int, default=0)
    ap.add_argument("--out", default=None, help="output dir (default demo/out/worldgen)")
    args = ap.parse_args()

    cfg = load_config()
    world = generate_world(cfg, args.world_id)
    print(f"world {args.world_id}: {world.n}x{world.n} cells @ {world.cell_m} m, "
          f"{len(world.town.buildings)} buildings, {len(world.town.exits)} exits, "
          f"wind {world.wind_base_speed_ms:.1f} m/s toward {world.wind_base_dir_deg:.0f} deg")

    if args.preview:
        out = pathlib.Path(args.out) if args.out else repo_root() / "demo" / "out" / "worldgen"
        out.mkdir(parents=True, exist_ok=True)
        save_terrain_png(world, out / "terrain.png")
        save_fuel_png(world, out / "fuel.png")
        save_town_png(world, out / "town.png")
        print(f"previews saved to {out}")


if __name__ == "__main__":
    main()
