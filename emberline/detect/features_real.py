"""Windows + features for the REAL Kaggle smoke dataset. Real data only.

All bytes come through emberline.data (manifest + guard); nothing here can
read synthetic data, and no synthetic-trained checkpoint is loaded anywhere
in the *_real modules.

Leak handling (measured in data/real/INVENTORY_PRELIM.md, non-negotiable):
THREE columns leak label information through recording order — the unnamed
row index (76.4% accuracy alone), UTC, and CNT (90.0% alone; no negative row
has CNT > 5,743). CNT's exactly-4 reset points define the 5 ground-truth
sessions (already applied by ingest; sessions are the split unit in
MANIFEST.yaml). After that the three columns are dropped: `RETAINED_CHANNELS`
below is the complete feature surface, and tests assert both the drop and
that no retained feature correlates with row order.

Window design: 60 s windows at the native 1 Hz. Train stride 10 s —
adjacent 1 s-stride windows share 59/60 samples and add almost no
information to a GBM/CNN fit, so 10 s keeps full coverage with 6x fewer
near-duplicate rows. Val/test stride 1 s — evaluation must mirror
deployment, where the node classifies every second, and detection latency
is only meaningful at 1 s resolution. A window's label is the label of its
LAST sample: the decision instant. Windows never cross session boundaries.

Gas channels and the TVOC caveats (measured):
* TVOC polarity is INVERTED vs our node physics: mean 4,596.6 ppb when
  Fire Alarm = 0 vs 882.0 when = 1 — higher WITHOUT alarm. Raw gas
  resistance on our node falls in smoke; this vendor index runs the other
  way for whatever this label actually marks. The log-ratio feature is
  therefore signed-free: the model learns the direction from data, nothing
  hard-codes "up means fire".
* 4.3% of TVOC samples are exactly 0 (2,698 rows, straddling both classes:
  2,038 negative / 660 positive). They are never dropped. The log uses
  EPS = 1.0 in numerator and denominator, so a zero reading gives
  log(eps/(baseline+eps)) — a large negative but finite value.
* Rolling-baseline-zero: the baseline is the causal rolling median of the
  preceding 300 s within the session (min 1 sample). If the baseline
  median is itself 0 (a stretch of zero readings), EPS floors the
  denominator. For the first window of a session (no history) the baseline
  is the window's own mean, making the feature exactly 0 — "no evidence of
  change" rather than a fabricated excursion.
* These vendor-index channels (TVOC, and Raw H2 / Raw Ethanol which are
  raw ADC-ish gas signals) are NOT a drop-in proxy for the node's
  gas_ohms; that is precisely why our own logged sessions are the next
  step. eCO2 is excluded entirely: it is derived by the same vendor
  algorithm from the same sensing element as TVOC and adds a second copy
  of the same index.
"""

from __future__ import annotations


import numpy as np
import pandas as pd
import yaml

from emberline.data import REAL_DATA_ROOT, assert_real_data_path

WINDOW = 60          # seconds @ 1 Hz
TRAIN_STRIDE = 10    # see module docstring
EVAL_STRIDE = 1
GAS_EPS = 1.0        # log-ratio epsilon, units of the raw channel
BASELINE_S = 300     # causal rolling-baseline horizon, seconds

LEAK_COLUMNS = ("row_index", "UTC", "CNT")
LABEL = "Fire Alarm"

# The complete feature surface. Order fixed: it defines CNN channel order.
#
# Temperature[C] is EXCLUDED as a fabricated channel (v2 decision, measured):
# sessions 1 and 4 are exact row-wise copies of sessions 0 and 3 in every
# other channel INCLUDING the label, while Temperature differs by large,
# non-constant, physically implausible offsets (mean +16.9 with swings to
# +49 between s0->s1; mean -27.6 with swings to -81.6 degC between s3->s4).
# The publisher evidently duplicated two recordings with altered temperature
# traces. Consistently, the channel's class direction is incoherent: AUC
# 0.027 on the train sessions (cold=fire) vs 0.990 on val (warm=fire) —
# measured before any test evaluation. A channel with fabricated values and
# contradictory direction is not sensor data; no model here gets it.
# Pressure[hPa] is EXCLUDED by the row-order leak test: its window mean has
# |spearman| 0.937 vs within-session row order on the TRAIN sessions —
# barometric level is a slow weather drift that acts as a session clock, the
# exact proxy the leak test exists to catch (ceiling 0.5; bookkeeping leaks
# like CNT measure ~1.0). Removed from the whole surface, not just the GBM:
# the CNN could trivially recover the same clock from a raw pressure channel.
RETAINED_CHANNELS: tuple[str, ...] = (
    "Humidity[%]",
    "TVOC[ppb]",
    "Raw H2",
    "Raw Ethanol",
    "PM1.0",
    "PM2.5",
    "NC0.5",
    "NC1.0",
    "NC2.5",
)

GAS_CHANNELS = ("TVOC[ppb]", "Raw H2", "Raw Ethanol")
MEAN_MAX_SLOPE = ("PM1.0", "PM2.5", "NC0.5", "NC1.0", "NC2.5")
# Temperature and Pressure means are gone with their channels (notes at
# RETAINED_CHANNELS). Humidity mean stays: |spearman| vs row order 0.42,
# under the 0.5 leak ceiling — though its class direction is also unstable
# across sessions (train->val AUC 0.77->0.42); the report carries the caveat.
MEAN_ONLY = ("Humidity[%]",)
# Raw H2 / Raw Ethanol are gas channels: they get the causal log-ratio (like
# TVOC) plus window slope — dynamics, not levels. Their absolute mean/max are
# deliberately excluded: both drift near-monotonically over a session
# (measured |spearman| vs row order 0.92), which makes a level feature a
# within-session clock proxy, the exact failure the leak test guards against.
SLOPE_ONLY = ("Raw H2", "Raw Ethanol")


def load_manifest() -> dict:
    path = REAL_DATA_ROOT / "MANIFEST.yaml"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} missing — run `python -m emberline.data.ingest` first"
        )
    return yaml.safe_load(path.read_text())


def load_sessions(split: str, manifest: dict | None = None) -> list[pd.DataFrame]:
    """Load the interim parquet of every smoke session assigned to *split*."""
    manifest = manifest or load_manifest()
    groups = manifest["datasets"]["kaggle_smoke"]["groups"]
    out = []
    for g in groups:
        if g["split"] != split:
            continue
        p = assert_real_data_path(REAL_DATA_ROOT / g["interim_path"])
        df = pd.read_parquet(p)
        for col in LEAK_COLUMNS:
            if col not in df.columns:
                raise RuntimeError(f"{p}: expected raw column {col!r} in interim data")
        out.append(df)
    if not out:
        raise RuntimeError(f"no smoke sessions assigned to split {split!r}")
    return out


def session_windows(df: pd.DataFrame, stride: int) -> dict[str, np.ndarray]:
    """Slice one session into (C, 60) windows.

    Returns dict with:
      x     (N, C, 60) float32 raw channels (RETAINED_CHANNELS order)
      y     (N,) int8, label of the window's last sample
      end   (N,) int32, row offset of the last sample within the session
            (1 Hz -> also seconds since session start; UTC itself is a leak
            column and is never used for timing)
      gas_logratio (N, len(GAS_CHANNELS)) float32
    """
    n = len(df)
    if n < WINDOW:
        raise RuntimeError(f"session shorter ({n}) than one window ({WINDOW})")
    chans = df[list(RETAINED_CHANNELS)].to_numpy(dtype=np.float32).T  # (C, n)
    labels = df[LABEL].to_numpy(dtype=np.int8)
    starts = np.arange(0, n - WINDOW + 1, stride, dtype=np.int64)
    ends = starts + WINDOW - 1

    idx = starts[:, None] + np.arange(WINDOW)[None, :]         # (N, 60)
    x = chans[:, idx].transpose(1, 0, 2)                        # (N, C, 60)
    y = labels[ends]
    # entirely fire-free windows: the denominator of FP-per-node-day
    clean = labels[idx].max(axis=1) == 0

    gas = np.empty((len(starts), len(GAS_CHANNELS)), dtype=np.float32)
    for j, ch in enumerate(GAS_CHANNELS):
        s = df[ch].astype("float64")
        # causal rolling median of the preceding BASELINE_S samples:
        # value at row r summarises rows (r-BASELINE_S, r]; we read it at
        # row start-1 so the baseline never sees the window itself.
        roll = s.rolling(BASELINE_S, min_periods=1).median().to_numpy()
        base = np.where(starts > 0, roll[np.maximum(starts - 1, 0)], np.nan)
        wmean = s.to_numpy()[idx].mean(axis=1)
        base = np.where(np.isnan(base), wmean, base)  # first window: ratio 1 -> 0
        gas[:, j] = np.log((wmean + GAS_EPS) / (base + GAS_EPS))

    return {"x": x, "y": y, "end": ends.astype(np.int32), "gas_logratio": gas,
            "clean": clean}


FEATURE_NAMES: list[str] = (
    [f"{c}:{s}" for c in MEAN_MAX_SLOPE for s in ("mean", "max", "slope")]
    + [f"{c}:mean" for c in MEAN_ONLY]
    + [f"{c}:slope" for c in SLOPE_ONLY]
    + [f"{c}:logratio" for c in GAS_CHANNELS]
)


def window_features(w: dict[str, np.ndarray]) -> np.ndarray:
    """(N, C, 60) windows -> (N, F) GBM features, per the docstring contract."""
    x = w["x"]
    t = np.arange(WINDOW, dtype=np.float32)
    t = (t - t.mean()) / t.std()
    ci = {c: i for i, c in enumerate(RETAINED_CHANNELS)}
    feats = []
    for c in MEAN_MAX_SLOPE:
        ch = x[:, ci[c], :]
        feats += [ch.mean(1), ch.max(1), (ch * t).mean(1)]
    for c in MEAN_ONLY:
        feats.append(x[:, ci[c], :].mean(1))
    for c in SLOPE_ONLY:
        feats.append((x[:, ci[c], :] * t).mean(1))
    for j in range(len(GAS_CHANNELS)):
        feats.append(w["gas_logratio"][:, j])
    out = np.stack(feats, axis=1).astype(np.float32)
    assert out.shape[1] == len(FEATURE_NAMES)
    return out


def split_windows(split: str, stride: int, manifest: dict | None = None):
    """All windows of a split, concatenated, with session ids kept."""
    manifest = manifest or load_manifest()
    groups = [g for g in manifest["datasets"]["kaggle_smoke"]["groups"] if g["split"] == split]
    xs, ys, ends, gas, sids, clean = [], [], [], [], [], []
    for g in groups:
        p = assert_real_data_path(REAL_DATA_ROOT / g["interim_path"])
        df = pd.read_parquet(p)
        w = session_windows(df, stride)
        xs.append(w["x"]); ys.append(w["y"]); ends.append(w["end"])
        gas.append(w["gas_logratio"]); clean.append(w["clean"])
        sids.append(np.full(len(w["y"]), g["session_id"], dtype=np.int16))
    return {
        "x": np.concatenate(xs),
        "y": np.concatenate(ys),
        "end": np.concatenate(ends),
        "gas_logratio": np.concatenate(gas),
        "session": np.concatenate(sids),
        "clean": np.concatenate(clean),
    }
