"""Ingest real datasets: validate, normalize to parquet, write the manifest.

    python -m emberline.data.ingest [--raw-dir data/real/raw]

Walks ``data/real/raw/``, validates every known file against its schema in
:mod:`emberline.data.schemas`, normalizes the smoke CSV to one parquet per
recording session in ``data/real/interim/``, and writes
``data/real/MANIFEST.yaml`` recording path, SHA-256, bytes, source id,
licence note and the SPLIT assignment of every training group.

Idempotent: a second run recomputes everything deterministically and writes
byte-identical manifest content (SHA-256 of raw files is cached by
(size, mtime_ns) in ``interim/.hash_cache.json`` so re-runs are fast; the
cache never feeds the manifest with a stale digest for a changed file).

Split policy (assigned once, deterministically, and recorded here):

* Kaggle smoke — split BY SESSION, never by row/window. Sessions are the
  ground-truth recording units, derived from the CNT counter's exactly-4
  reset points. Assignment (fixed):
      session 0 (24,994 rows, 87.3% pos)  -> train
      session 1 (24,994 rows, 87.3% pos)  -> val   (threshold selection)
      session 2 ( 1,154 rows, 97.1% pos)  -> test  (fire session: latency)
      session 3 ( 5,744 rows,  0.0% pos)  -> train (fire-free)
      session 4 ( 5,744 rows,  0.0% pos)  -> test  (fire-free: FP/node-day)
  Rationale: every split contains positives or a fire-free span as its role
  requires; test gets one fire-heavy session for detection latency and one
  fire-free session for false positives per node-day; nothing is tuned on it.
* NDWS — the dataset ships its own 15/2/2 shard split in the filenames
  (train/eval/test). Respected as-is; never reshuffled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from emberline.data import REAL_DATA_ROOT, assert_real_data_path
from emberline.data.schemas import (
    KAGGLE_SMOKE,
    NDWS_TILE_LEN,
    validate_ndws_example,
)
from emberline.data.tfrecord_lite import iter_examples, iter_tfrecord

# Fixed, recorded-once split assignment for the smoke sessions (see module
# docstring for the rationale). Changing this invalidates every downstream
# result and must never happen silently: the manifest carries a fingerprint
# of each session and tests assert this exact mapping.
SMOKE_SESSION_SPLITS: dict[int, str] = {0: "train", 1: "val", 2: "test", 3: "train", 4: "test"}

SOURCES = {
    "kaggle_smoke": {
        "source_id": "kaggle:deepcontractor/smoke-detection-dataset",
        "license_note": (
            "Kaggle card not yet captured into data/real/external/; "
            "internal training/evaluation use only until the licence text is on file."
        ),
    },
    "ndws": {
        "source_id": "kaggle:fantineh/next-day-wildfire-spread (Huot et al. 2022, arXiv:2112.02447)",
        "license_note": (
            "Commonly cited as CC BY 4.0; licence text not yet captured into "
            "data/real/external/ — confirm before redistribution."
        ),
    },
}


def sha256_file(path: Path, cache: dict | None = None) -> str:
    """SHA-256 of a file, with an optional (size, mtime_ns) keyed cache."""
    st = path.stat()
    key = str(path.as_posix())
    if cache is not None:
        hit = cache.get(key)
        if hit and hit["bytes"] == st.st_size and hit["mtime_ns"] == st.st_mtime_ns:
            return hit["sha256"]
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    digest = h.hexdigest()
    if cache is not None:
        cache[key] = {"bytes": st.st_size, "mtime_ns": st.st_mtime_ns, "sha256": digest}
    return digest


def derive_smoke_sessions(df: pd.DataFrame) -> list[tuple[int, int]]:
    """Session (start, end) row ranges from CNT reset points."""
    cnt = df["CNT"].to_numpy()
    resets = np.where(np.diff(cnt) < 0)[0] + 1
    bounds = [0, *resets.tolist(), len(df)]
    return [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1)]


def ingest_smoke(csv_path: Path, interim_dir: Path, cache: dict) -> dict:
    """Validate the smoke CSV, write one parquet per session, return manifest entry."""
    assert_real_data_path(csv_path)
    with open(csv_path, encoding="utf-8", newline="") as f:
        header = f.readline().rstrip("\r\n").split(",")
    KAGGLE_SMOKE.validate_columns(header)

    df = pd.read_csv(csv_path)
    # pandas names the empty-header column 'Unnamed: 0'; keep bytes-true
    # content but give interim files stable, addressable column names.
    df = df.rename(columns={"Unnamed: 0": "row_index"})

    out_dir = interim_dir / "kaggle_smoke"
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = []
    for sid, (a, b) in enumerate(derive_smoke_sessions(df)):
        sdf = df.iloc[a:b].reset_index(drop=True)
        sdf.insert(0, "session_id", sid)
        out = out_dir / f"session_{sid:02d}.parquet"
        sdf.to_parquet(out, index=False)
        labels = sdf["Fire Alarm"].to_numpy()
        try:
            rel = out.relative_to(REAL_DATA_ROOT)
        except ValueError:  # test fixtures use an out-of-tree interim dir
            rel = out.relative_to(interim_dir)
        groups.append(
            {
                "session_id": sid,
                "split": SMOKE_SESSION_SPLITS[sid],
                "interim_path": str(rel.as_posix()),
                "rows": int(len(sdf)),
                "positives": int(labels.sum()),
                "utc_first": int(sdf["UTC"].iloc[0]),
                "utc_last": int(sdf["UTC"].iloc[-1]),
                "row_range": [int(a), int(b)],
            }
        )
    if len(groups) != 5:
        raise RuntimeError(
            f"expected exactly 5 smoke sessions (4 CNT resets), found {len(groups)}; "
            "the raw file changed — re-inventory before ingesting"
        )
    st = csv_path.stat()
    return {
        "path": str(csv_path.relative_to(REAL_DATA_ROOT).as_posix()),
        "bytes": int(st.st_size),
        "sha256": sha256_file(csv_path, cache),
        "format": "csv",
        **SOURCES["kaggle_smoke"],
        "split_policy": "by session id (CNT resets); assigned once, recorded below",
        "groups": groups,
    }


_NDWS_SPLIT_RE = re.compile(r"next_day_wildfire_spread_(train|eval|test)_(\d+)\.tfrecord$")


def ingest_ndws(shard_paths: list[Path], cache: dict) -> list[dict]:
    """Validate NDWS shards (spec of first example + framing scan), return entries."""
    entries = []
    for p in sorted(shard_paths):
        assert_real_data_path(p)
        m = _NDWS_SPLIT_RE.search(p.name)
        if not m:
            raise RuntimeError(f"unrecognized NDWS shard name: {p.name}")
        split = {"train": "train", "eval": "val", "test": "test"}[m.group(1)]
        first = next(iter_examples(p))
        validate_ndws_example(list(first), {k: len(v) for k, v in first.items()})
        n_records = sum(1 for _ in iter_tfrecord(p, verify_crc=False))
        st = p.stat()
        entries.append(
            {
                "path": str(p.relative_to(REAL_DATA_ROOT).as_posix()),
                "bytes": int(st.st_size),
                "sha256": sha256_file(p, cache),
                "format": "tfrecord",
                **SOURCES["ndws"],
                "split": split,
                "split_policy": "dataset's own shipped shard split (15 train / 2 eval / 2 test); never reshuffled",
                "records": int(n_records),
                "tile": f"64x64 ({NDWS_TILE_LEN} floats/feature)",
            }
        )
    return entries


def build_manifest(raw_dir: Path, interim_dir: Path) -> dict:
    raw_dir = assert_real_data_path(raw_dir)
    cache_path = interim_dir / ".hash_cache.json"
    cache: dict = {}
    if cache_path.exists():
        cache = json.loads(cache_path.read_text())

    smoke_csv = raw_dir / "kaggle_smoke" / "smoke_detection_iot.csv"
    ndws_shards = sorted((raw_dir / "ndws").glob("*.tfrecord"))
    known = {smoke_csv, *ndws_shards}
    all_files = {p for p in raw_dir.rglob("*") if p.is_file() and p.name != ".gitkeep"}
    unknown = sorted(all_files - known)
    if unknown:
        raise RuntimeError(
            "raw/ contains files no schema covers (refusing to ingest blind):\n  "
            + "\n  ".join(str(p) for p in unknown)
        )

    manifest = {
        "manifest_version": 1,
        "root": "data/real",
        "note": (
            "Written by python -m emberline.data.ingest. Identity is SHA-256, "
            "never size (16 of 19 NDWS shards are byte-identical in size). "
            "Split assignments here are authoritative and assigned once."
        ),
        "datasets": {
            "kaggle_smoke": ingest_smoke(smoke_csv, interim_dir, cache),
            "ndws": {
                "split_counts": {"train": 15, "val": 2, "test": 2},
                "shards": ingest_ndws(ndws_shards, cache),
            },
        },
    }
    interim_dir.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(cache, indent=1, sort_keys=True))
    return manifest


def manifest_yaml(manifest: dict) -> str:
    return yaml.safe_dump(manifest, sort_keys=True, default_flow_style=False, width=100)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw-dir", default=str(REAL_DATA_ROOT / "raw"))
    ap.add_argument("--interim-dir", default=str(REAL_DATA_ROOT / "interim"))
    ap.add_argument("--manifest", default=str(REAL_DATA_ROOT / "MANIFEST.yaml"))
    args = ap.parse_args(argv)

    manifest = build_manifest(Path(args.raw_dir), Path(args.interim_dir))
    text = manifest_yaml(manifest)
    out = Path(args.manifest)
    if out.exists() and out.read_text() == text:
        print(f"manifest unchanged: {out}")
    else:
        out.write_text(text)
        print(f"wrote {out}")
    smoke = manifest["datasets"]["kaggle_smoke"]
    print(f"smoke: {len(smoke['groups'])} sessions -> "
          + ", ".join(f"s{g['session_id']}:{g['split']}" for g in smoke["groups"]))
    shards = manifest["datasets"]["ndws"]["shards"]
    print(f"ndws: {len(shards)} shards, {sum(s['records'] for s in shards)} records")
    return 0


if __name__ == "__main__":
    sys.exit(main())
