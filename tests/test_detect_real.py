"""Tests for the real-data smoke detection pipeline (Phase 2).

The heart of this file is the leak guard: the three measured leak columns
(row index, UTC, CNT) must be gone from the feature surface, and NO retained
feature may correlate with row order.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from emberline.data import REAL_DATA_ROOT
from emberline.detect.features_real import (
    FEATURE_NAMES,
    GAS_CHANNELS,
    LEAK_COLUMNS,
    RETAINED_CHANNELS,
    TRAIN_STRIDE,
    WINDOW,
    session_windows,
    split_windows,
    window_features,
)

REAL_READY = (REAL_DATA_ROOT / "MANIFEST.yaml").exists() and (
    REAL_DATA_ROOT / "interim" / "kaggle_smoke" / "session_00.parquet"
).exists()

# |spearman| ceiling for every retained feature vs within-session row order.
# Stated rationale: bookkeeping leaks are near-perfect rank correlates of row
# order (CNT and the row index measure ~1.0); ambient *level* features that
# drift with session time measured up to 0.94 (pressure mean) and were
# removed or capped; the worst surviving legitimate feature measured 0.42
# (particulate means — fires cluster in session time, so some correlation is
# physics, not leakage). 0.5 splits the two regimes with margin on both sides.
SPEARMAN_CEILING = 0.5


def _fixture_session(n=300, seed=0, tvoc_zeros=False) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.uniform(1, 100, n) for c in RETAINED_CHANNELS})
    if tvoc_zeros:
        df.loc[: n // 3, "TVOC[ppb]"] = 0.0
    df["Fire Alarm"] = (rng.random(n) < 0.5).astype(int)
    # leak columns present in interim data, but never read by the featureizer
    df["row_index"] = np.arange(n)
    df["UTC"] = 1_654_000_000 + np.arange(n)
    df["CNT"] = np.arange(n)
    return df


# ---------------------------------------------------------------------------
# Leak columns are out of the feature surface
# ---------------------------------------------------------------------------

def test_leak_columns_not_retained():
    for col in LEAK_COLUMNS:
        assert col not in RETAINED_CHANNELS
        assert not any(col in f for f in FEATURE_NAMES)
    # the fabricated temperature channel is out too (see features_real.py)
    assert "Temperature[C]" not in RETAINED_CHANNELS


def test_features_only_read_retained_channels():
    df = _fixture_session()
    w = session_windows(df, stride=10)
    assert w["x"].shape[1] == len(RETAINED_CHANNELS)
    df2 = df.copy()
    df2["CNT"] = df2["CNT"][::-1].to_numpy()  # mangle a leak column
    df2["UTC"] = 0
    df2["row_index"] = 7
    w2 = session_windows(df2, stride=10)
    assert np.array_equal(w["x"], w2["x"])  # features blind to leak columns
    assert np.array_equal(window_features(w), window_features(w2))


@pytest.mark.skipif(not REAL_READY, reason="real interim data not present")
def test_no_retained_feature_correlates_with_row_order():
    from scipy.stats import spearmanr

    w = split_windows("train", TRAIN_STRIDE)
    F = window_features(w)
    worst = {}
    for j, name in enumerate(FEATURE_NAMES):
        rhos = []
        for s in np.unique(w["session"]):
            m = w["session"] == s
            if F[m, j].std() == 0:
                continue
            rho, _ = spearmanr(F[m, j], w["end"][m])
            rhos.append(abs(float(rho)))
        worst[name] = max(rhos) if rhos else 0.0
    offenders = {k: round(v, 3) for k, v in worst.items() if v >= SPEARMAN_CEILING}
    assert not offenders, (
        f"features correlate with row order (|spearman| >= {SPEARMAN_CEILING}): "
        f"{offenders} — a row-order proxy slipped back into the feature surface"
    )


# ---------------------------------------------------------------------------
# Window mechanics
# ---------------------------------------------------------------------------

def test_window_label_is_last_sample_and_no_session_crossing():
    df = _fixture_session(n=200, seed=3)
    w = session_windows(df, stride=7)
    labels = df["Fire Alarm"].to_numpy()
    assert np.array_equal(w["y"], labels[w["end"]])
    assert w["end"].max() == len(df) - 1 - ((len(df) - WINDOW) % 7)
    # split_windows builds windows per session parquet, so no window can
    # span two sessions by construction; assert the shape contract instead.
    assert w["x"].shape == (len(w["y"]), len(RETAINED_CHANNELS), WINDOW)


def test_gas_logratio_zero_and_first_window_rules():
    df = _fixture_session(n=400, seed=5, tvoc_zeros=True)
    w = session_windows(df, stride=5)
    gas = w["gas_logratio"]
    assert np.all(np.isfinite(gas)), "epsilon must keep log-ratio finite on zeros"
    # first window has no history: baseline falls back to the window's own
    # mean, so the feature is exactly 0 for every gas channel
    assert np.allclose(gas[0], 0.0, atol=1e-6)
    assert gas.shape[1] == len(GAS_CHANNELS)


@pytest.mark.skipif(not REAL_READY, reason="real interim data not present")
def test_real_tvoc_zeros_are_kept():
    # 2,698 exact-zero TVOC rows straddle both classes; they must survive
    # ingest and windowing (never dropped).
    total = 0
    for split in ("train", "val", "test"):
        for df in _load_sessions(split):
            total += int((df["TVOC[ppb]"] == 0).sum())
    assert total == 2698


def _load_sessions(split):
    from emberline.detect.features_real import load_sessions

    return load_sessions(split)
