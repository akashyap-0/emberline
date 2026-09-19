"""NDWS shard conversion + loading. Real data only.

    python -m emberline.data.ndws          # convert all shards (resumable)

Converts each raw tfrecord shard into flat arrays under
``data/real/interim/ndws/`` so training never re-parses protos:

    <shard stem>.x.npy   (N, 12, 64, 64) float16 — 11 covariates + PrevFireMask
    <shard stem>.y.npy   (N, 64, 64)     int8    — FireMask: 1 fire, 0 no fire,
                                                   -1 uncertain (EXCLUDED from
                                                   loss and metrics downstream)

float16 for the inputs: every NDWS covariate fits comfortably in half
precision (largest magnitudes: NDVI ~1e4, Raw population ~1e3) and the
training path standardizes to float32 anyway; halving the bytes keeps the
whole train split resident in RAM. Masks are exact int8.

Conversion is per-shard resumable: a shard whose output files already exist
and match the expected record count is skipped, so an interrupted run picks
up where it stopped. Channel order is NDWS_COVARIATES + PrevFireMask.

The split of every shard comes from data/real/MANIFEST.yaml (the dataset's
own shipped 15/2/2 split; never reshuffled).
"""

from __future__ import annotations

import sys

import numpy as np
import yaml

from emberline.data import REAL_DATA_ROOT, assert_real_data_path
from emberline.data.schemas import (
    NDWS_COVARIATES,
    NDWS_INPUT_MASK,
    NDWS_TARGET,
    NDWS_TILE_SIDE,
    validate_ndws_example,
)
from emberline.data.tfrecord_lite import iter_examples

INTERIM = REAL_DATA_ROOT / "interim" / "ndws"
CHANNELS: tuple[str, ...] = NDWS_COVARIATES + (NDWS_INPUT_MASK,)
S = NDWS_TILE_SIDE

# Measured artifact (train shards): the wind-direction channel `th` carries
# garbage fill values far outside [0, 360] deg (observed min -465,922), which
# overflow float16 to inf. Conversion clips `th` to its physical range and
# clamps every covariate to the float16-finite range; other channels' rare
# implausible-but-finite values (negative erc/pr, 0 K temperatures) are left
# for standardization to dampen — cleaning beyond physics bounds would be
# modelling, which does not belong in interim data.
F16_LIMIT = 60000.0
CHANNEL_CLIP: dict[str, tuple[float, float]] = {"th": (0.0, 360.0)}


def convert_shard(raw_path, expected_records: int) -> tuple[str, int]:
    """Convert one tfrecord shard to .x.npy/.y.npy; skip if already done."""
    raw_path = assert_real_data_path(raw_path)
    stem = raw_path.stem
    xp = INTERIM / f"{stem}.x.npy"
    yp = INTERIM / f"{stem}.y.npy"
    if xp.exists() and yp.exists():
        n = np.load(xp, mmap_mode="r").shape[0]
        if n == expected_records:
            return "skipped", n
    xs = np.empty((expected_records, len(CHANNELS), S, S), dtype=np.float16)
    ys = np.empty((expected_records, S, S), dtype=np.int8)
    n = 0
    for ex in iter_examples(raw_path):
        validate_ndws_example(list(ex), {k: len(v) for k, v in ex.items()})
        for c, name in enumerate(CHANNELS):
            lo, hi = CHANNEL_CLIP.get(name, (-F16_LIMIT, F16_LIMIT))
            xs[n, c] = np.clip(ex[name].reshape(S, S), lo, hi).astype(np.float16)
        ys[n] = ex[NDWS_TARGET].reshape(S, S).astype(np.int8)
        n += 1
    if n != expected_records:
        raise RuntimeError(f"{raw_path}: parsed {n} records, manifest says {expected_records}")
    INTERIM.mkdir(parents=True, exist_ok=True)
    np.save(xp, xs)
    np.save(yp, ys)
    return "converted", n


def shard_table() -> list[dict]:
    """Manifest NDWS shard entries with resolved interim paths."""
    manifest = yaml.safe_load((REAL_DATA_ROOT / "MANIFEST.yaml").read_text())
    out = []
    for s in manifest["datasets"]["ndws"]["shards"]:
        raw = REAL_DATA_ROOT / s["path"]
        stem = raw.stem
        out.append({
            "raw": raw, "split": s["split"], "records": s["records"],
            "x": INTERIM / f"{stem}.x.npy", "y": INTERIM / f"{stem}.y.npy",
        })
    return out


def load_split(split: str, mmap: bool = True):
    """(x, y) for a split, concatenated over its converted shards.

    x: (N, 12, 64, 64) float16, y: (N, 64, 64) int8. With mmap=True the
    per-shard arrays stay on disk and only touched batches are paged in.
    """
    xs, ys = [], []
    for s in shard_table():
        if s["split"] != split:
            continue
        assert_real_data_path(s["x"])
        if not s["x"].exists():
            raise FileNotFoundError(
                f"{s['x']} missing — run `python -m emberline.data.ndws` first")
        xs.append(np.load(s["x"], mmap_mode="r" if mmap else None))
        ys.append(np.load(s["y"], mmap_mode="r" if mmap else None))
    if not xs:
        raise RuntimeError(f"no NDWS shards for split {split!r}")
    return xs, ys


def main(argv=None) -> int:
    total = 0
    for s in shard_table():
        status, n = convert_shard(s["raw"], s["records"])
        total += n
        print(f"{s['raw'].name}: {status} ({n} tiles, split={s['split']})")
    print(f"total {total} tiles in {INTERIM}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
