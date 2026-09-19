"""Explicit column/feature schemas for every real dataset we ingest.

Three contracts live here:

1. KAGGLE_SMOKE — the Kaggle smoke-detection CSV exactly as it is on disk,
   leak columns and all. The schema describes reality; deciding what to
   *drop* is the feature builder's job, not the validator's.
2. NDWS — the tf.train.Example feature spec of the Next Day Wildfire Spread
   tfrecords (64x64 float tiles per key).
3. ESP32_LOGGER — the future contract for our own node's CSV logger.
   Schema only tonight: no data exists yet, but ingest must be drop-in
   ready when the first session is logged.

Validators REJECT unknown and missing columns with messages that say what
was found and what was expected. They never coerce, rename, or fill.
"""

from __future__ import annotations

from dataclasses import dataclass, field


class SchemaError(ValueError):
    """A dataset's columns do not match its declared schema."""


@dataclass(frozen=True)
class TableSchema:
    """An exact, ordered column contract for a delimited table."""

    name: str
    columns: tuple[str, ...]
    notes: dict[str, str] = field(default_factory=dict)

    def validate_columns(self, found: list[str] | tuple[str, ...]) -> None:
        """Reject any deviation from the declared column set/order."""
        found = list(found)
        expected = list(self.columns)
        if found == expected:
            return
        missing = [c for c in expected if c not in found]
        unknown = [c for c in found if c not in expected]
        msgs = [f"{self.name}: header does not match the declared schema."]
        if missing:
            msgs.append(f"  missing columns: {missing!r}")
        if unknown:
            msgs.append(f"  unknown columns: {unknown!r}")
        if not missing and not unknown:
            msgs.append(
                f"  same columns but wrong order:\n    found    {found!r}\n    expected {expected!r}"
            )
        msgs.append(
            "  This validator never coerces or renames. If the source file "
            "changed, update the schema deliberately (and the manifest), do "
            "not patch the data."
        )
        raise SchemaError("\n".join(msgs))


# ---------------------------------------------------------------------------
# 1. Kaggle smoke-detection CSV — as it truly is on disk.
# ---------------------------------------------------------------------------
# The first column has an EMPTY name: it is the exporter's unnamed row index.
# It, UTC and CNT are measured leak columns (see data/real/INVENTORY_PRELIM.md)
# but they are part of the file, so they are part of the schema; dropping them
# is a modelling decision made downstream (emberline/detect/features_real.py).
KAGGLE_SMOKE = TableSchema(
    name="kaggle_smoke/smoke_detection_iot.csv",
    columns=(
        "",  # unnamed exporter row index — leak column, dropped by features
        "UTC",  # epoch seconds — leak column, dropped by features
        "Temperature[C]",
        "Humidity[%]",
        "TVOC[ppb]",  # vendor VOC index; polarity INVERTED vs raw gas_ohms
        "eCO2[ppm]",  # vendor-derived index
        "Raw H2",
        "Raw Ethanol",
        "Pressure[hPa]",
        "PM1.0",
        "PM2.5",
        "NC0.5",
        "NC1.0",
        "NC2.5",
        "CNT",  # sample counter; its 4 resets define the 5 sessions, then dropped
        "Fire Alarm",  # label; 71.5% positive — nothing like deployment priors
    ),
    notes={
        "": "unnamed row index 0..62629; threshold on it alone scores 76.4% (majority 71.5%)",
        "CNT": "threshold on it alone scores 90.0%; no negative row has CNT > 5743",
        "UTC": "epoch seconds, separates sessions by wall clock",
        "TVOC[ppb]": "mean 4596.6 when Fire Alarm=0 vs 882.0 when =1 (inverted); 4.3% exact zeros",
    },
)


# ---------------------------------------------------------------------------
# 2. NDWS tfrecord feature spec.
# ---------------------------------------------------------------------------
# Every tf.train.Example carries 13 float lists of length 64*64 = 4096:
# 11 covariate channels + prev_fire_mask input + FireMask target.
NDWS_TILE_SIDE = 64
NDWS_TILE_LEN = NDWS_TILE_SIDE * NDWS_TILE_SIDE

NDWS_COVARIATES: tuple[str, ...] = (
    "elevation",  # m
    "th",         # wind direction, deg
    "vs",         # wind speed, m/s
    "tmmn",       # min temperature, K
    "tmmx",       # max temperature, K
    "sph",        # specific humidity, kg/kg
    "pr",         # precipitation, mm
    "pdsi",       # Palmer drought severity index
    "NDVI",       # vegetation index
    "population", # people / km^2
    "erc",        # energy release component
)
NDWS_INPUT_MASK = "PrevFireMask"   # fire mask at day t
NDWS_TARGET = "FireMask"           # fire mask at day t+1
NDWS_FEATURES: tuple[str, ...] = NDWS_COVARIATES + (NDWS_INPUT_MASK, NDWS_TARGET)

# Mask semantics per the dataset spec (Huot et al. 2022):
#   1 = fire, 0 = no fire, -1 = uncertain/missing — EXCLUDED from loss and
#   from every metric downstream.
NDWS_MASK_FIRE = 1.0
NDWS_MASK_NOFIRE = 0.0
NDWS_MASK_UNCERTAIN = -1.0


def validate_ndws_example(feature_keys: list[str] | tuple[str, ...],
                          lengths: dict[str, int] | None = None) -> None:
    """Reject an NDWS example whose features deviate from the spec."""
    found = sorted(feature_keys)
    expected = sorted(NDWS_FEATURES)
    if found != expected:
        missing = [k for k in expected if k not in found]
        unknown = [k for k in found if k not in expected]
        raise SchemaError(
            "NDWS example does not match the declared feature spec.\n"
            f"  missing features: {missing!r}\n"
            f"  unknown features: {unknown!r}\n"
            "  Expected exactly 11 covariates + PrevFireMask + FireMask."
        )
    if lengths is not None:
        bad = {k: n for k, n in lengths.items() if n != NDWS_TILE_LEN}
        if bad:
            raise SchemaError(
                f"NDWS example has wrong-length features (expected {NDWS_TILE_LEN} "
                f"floats = 64x64): {bad!r}"
            )


# ---------------------------------------------------------------------------
# 3. Our future ESP32 logger contract — schema only tonight.
# ---------------------------------------------------------------------------
# One row per sample: millisecond uptime, particulates, raw gas resistance
# (ohms — NOT a vendor index; log(gas/baseline) goes NEGATIVE in smoke),
# temperature, relative humidity, pressure.
ESP32_LOGGER = TableSchema(
    name="esp32_logger",
    columns=(
        "ms",
        "pm1",
        "pm25",
        "pm10",
        "gas_ohms",
        "temp_c",
        "rh",
        "press_hpa",
    ),
    notes={
        "gas_ohms": "raw BME68x gas resistance in ohms; falls in smoke, "
        "unlike the Kaggle TVOC vendor index which runs the other way",
    },
)
