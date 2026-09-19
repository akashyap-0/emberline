"""Tests for the real-data infrastructure (emberline/data/).

Covers the four required guarantees:
  1. the no-synthetic-paths guard fires,
  2. schema validators reject unknown/missing columns,
  3. the manifest builder is deterministic,
  4. the split assignment is stable (and matches the committed manifest
     when the real data is present).
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import yaml

from emberline.data import REAL_DATA_ROOT, REPO_ROOT, SyntheticDataError, assert_real_data_path
from emberline.data.ingest import (
    SMOKE_SESSION_SPLITS,
    derive_smoke_sessions,
    ingest_smoke,
    manifest_yaml,
)
from emberline.data.schemas import ESP32_LOGGER, KAGGLE_SMOKE, SchemaError, validate_ndws_example

REAL_RAW_PRESENT = (REAL_DATA_ROOT / "raw" / "kaggle_smoke" / "smoke_detection_iot.csv").exists()


# ---------------------------------------------------------------------------
# 1. The no-synthetic guard
# ---------------------------------------------------------------------------

def test_guard_rejects_data_synthetic():
    with pytest.raises(SyntheticDataError):
        assert_real_data_path(REPO_ROOT / "data" / "synthetic" / "worlds" / "w0.npz")


def test_guard_rejects_any_nested_synthetic_dir(tmp_path):
    with pytest.raises(SyntheticDataError):
        assert_real_data_path(tmp_path / "data" / "synthetic" / "x.csv")


def test_guard_rejects_other_repo_data_dirs():
    # data/detect, data/surrogate etc. belong to the synthetic stack.
    with pytest.raises(SyntheticDataError):
        assert_real_data_path(REPO_ROOT / "data" / "detect" / "events.npz")
    with pytest.raises(SyntheticDataError):
        assert_real_data_path(REPO_ROOT / "data" / "surrogate" / "train.npz")


def test_guard_allows_real_and_scratch(tmp_path):
    assert_real_data_path(REAL_DATA_ROOT / "raw" / "kaggle_smoke" / "smoke_detection_iot.csv")
    assert_real_data_path(tmp_path / "fixture.csv")


# ---------------------------------------------------------------------------
# 2. Schema rejection
# ---------------------------------------------------------------------------

def test_smoke_schema_accepts_true_header():
    KAGGLE_SMOKE.validate_columns(list(KAGGLE_SMOKE.columns))


def test_smoke_schema_rejects_missing_column():
    cols = [c for c in KAGGLE_SMOKE.columns if c != "PM2.5"]
    with pytest.raises(SchemaError, match=r"missing columns.*PM2\.5"):
        KAGGLE_SMOKE.validate_columns(cols)


def test_smoke_schema_rejects_unknown_column():
    cols = list(KAGGLE_SMOKE.columns) + ["sneaky_extra"]
    with pytest.raises(SchemaError, match="unknown columns.*sneaky_extra"):
        KAGGLE_SMOKE.validate_columns(cols)


def test_smoke_schema_rejects_reordered_columns():
    cols = list(KAGGLE_SMOKE.columns)
    cols[2], cols[3] = cols[3], cols[2]
    with pytest.raises(SchemaError, match="wrong order"):
        KAGGLE_SMOKE.validate_columns(cols)


def test_esp32_schema_rejects_vendor_index_column():
    # The node contract is raw gas_ohms; a vendor TVOC column must be refused.
    cols = [c if c != "gas_ohms" else "TVOC[ppb]" for c in ESP32_LOGGER.columns]
    with pytest.raises(SchemaError):
        ESP32_LOGGER.validate_columns(cols)


def test_ndws_spec_rejects_missing_and_unknown_features():
    good = ["elevation", "th", "vs", "tmmn", "tmmx", "sph", "pr", "pdsi",
            "NDVI", "population", "erc", "PrevFireMask", "FireMask"]
    validate_ndws_example(good, {k: 4096 for k in good})
    with pytest.raises(SchemaError, match="missing"):
        validate_ndws_example([k for k in good if k != "FireMask"])
    with pytest.raises(SchemaError, match="unknown"):
        validate_ndws_example(good + ["bonus_channel"])
    with pytest.raises(SchemaError, match="wrong-length"):
        validate_ndws_example(good, {"elevation": 4095, **{k: 4096 for k in good[1:]}})


# ---------------------------------------------------------------------------
# 3. Manifest determinism (on a small fixture; no real data required)
# ---------------------------------------------------------------------------

def _write_smoke_fixture(path: Path, n_sessions: int = 3, rows_per: int = 40):
    rng = np.random.default_rng(7)
    with open(path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(KAGGLE_SMOKE.columns)
        idx = 0
        utc = 1_654_000_000
        for _s in range(n_sessions):
            for cnt in range(rows_per):
                w.writerow([
                    idx, utc, 20.0, 50.0,
                    float(rng.integers(0, 5000)), 400.0, 12000.0, 18000.0,
                    939.0, 1.0, 2.0, 100.0, 80.0, 20.0, cnt,
                    int(rng.random() < 0.5),
                ])
                idx += 1
                utc += 1
            utc += 1000


def test_smoke_ingest_deterministic_and_idempotent(tmp_path):
    raw = tmp_path / "raw"
    (raw / "kaggle_smoke").mkdir(parents=True)
    csv_path = raw / "kaggle_smoke" / "smoke_detection_iot.csv"
    _write_smoke_fixture(csv_path)

    # ingest_smoke requires exactly 5 sessions on the real file; here we call
    # the pieces it is built from to test determinism of the outputs.
    df = pd.read_csv(csv_path)
    s1 = derive_smoke_sessions(df)
    s2 = derive_smoke_sessions(df)
    assert s1 == s2 == [(0, 40), (40, 80), (80, 120)]


def test_manifest_yaml_is_stable():
    doc = {"b": 1, "a": {"z": [3, 2], "y": "x"}}
    assert manifest_yaml(doc) == manifest_yaml(dict(reversed(list(doc.items()))))


@pytest.mark.skipif(not REAL_RAW_PRESENT, reason="real raw data not on this machine")
def test_real_ingest_smoke_matches_committed_manifest(tmp_path):
    entry = ingest_smoke(
        REAL_DATA_ROOT / "raw" / "kaggle_smoke" / "smoke_detection_iot.csv",
        tmp_path, cache={},
    )
    committed = yaml.safe_load((REAL_DATA_ROOT / "MANIFEST.yaml").read_text())
    got = committed["datasets"]["kaggle_smoke"]
    assert entry["sha256"] == got["sha256"]
    for g_new, g_old in zip(entry["groups"], got["groups"]):
        assert g_new["session_id"] == g_old["session_id"]
        assert g_new["split"] == g_old["split"]
        assert g_new["rows"] == g_old["rows"]
        assert g_new["utc_first"] == g_old["utc_first"]


# ---------------------------------------------------------------------------
# 4. Split stability
# ---------------------------------------------------------------------------

def test_smoke_split_assignment_is_the_recorded_one():
    # Assigned once; changing this mapping invalidates every downstream
    # number and must show up as a failing test, not a silent drift.
    assert SMOKE_SESSION_SPLITS == {0: "train", 1: "val", 2: "test", 3: "train", 4: "test"}


@pytest.mark.skipif(not REAL_RAW_PRESENT, reason="real raw data not on this machine")
def test_committed_manifest_split_counts():
    committed = yaml.safe_load((REAL_DATA_ROOT / "MANIFEST.yaml").read_text())
    smoke = committed["datasets"]["kaggle_smoke"]
    assert [g["split"] for g in smoke["groups"]] == ["train", "val", "test", "train", "test"]
    shards = committed["datasets"]["ndws"]["shards"]
    from collections import Counter
    c = Counter(s["split"] for s in shards)
    assert c == {"train": 15, "val": 2, "test": 2}
    # NDWS keeps its shipped split: filename says what the split is.
    for s in shards:
        tag = {"train": "train", "val": "eval", "test": "test"}[s["split"]]
        assert f"_{tag}_" in s["path"]
