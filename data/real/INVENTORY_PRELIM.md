# Preliminary real-data inventory

Scan date: **2026-09-18**. Read-only survey: nothing was modified, moved,
renamed or ingested. This is a first look, not a manifest — no checksums were
computed and no file was parsed beyond what is noted below.

Method: `os.walk` over the raw tree; sizes from `stat()`; format sniffed from
the first 64 bytes (magic-number table, UTF-8 decode attempt as fallback). The
Kaggle CSV was read with the standard-library `csv` module to count rows and
list columns. The tfrecords were **not** opened beyond their first bytes.

## Where the data actually is

> **The repository's own `data/real/raw/` is empty** (skeleton `.gitkeep`
> only). The downloaded datasets live one directory **above** the repository:

| | path |
|---|---|
| repo raw dir (empty) | `conrad_challenge_fire_prototype/emberline/data/real/raw/` |
| **actual data** | `conrad_challenge_fire_prototype/data/real/raw/` |

That outer path is a sibling of the git repository and is not covered by its
`.gitignore`. The two sensible resolutions, both for the team to choose — no
files were touched:

1. **Move** the two dataset folders into `emberline/data/real/raw/`. The
   ignore rules added in this commit already exclude the bulk (verified with
   `git check-ignore`), so 3.7 GiB would stay out of git. This makes relative
   paths in future loaders simple.
2. **Leave them outside** the repo and point at them with an environment
   variable or a config key (e.g. `EMBERLINE_REAL_DATA_DIR`). This keeps a
   large download out of the project tree entirely and survives re-clones.

Until one is chosen, any code that assumes `data/real/raw/<dataset>` relative
to the repo root will find nothing.

## Summary

| | count | bytes | size |
|---|---|---|---|
| all files | 20 | 3,961,723,961 | 3.69 GiB |
| `kaggle_smoke/` | 1 | 5,834,376 | 5.6 MiB |
| `ndws/` | 19 | 3,955,889,585 | 3.68 GiB |

## File listing

Sizes in bytes and MiB, relative to the outer `data/real/raw/`.

| file | bytes | MiB |
|---|---|---|
| `kaggle_smoke/smoke_detection_iot.csv` | 5,834,376 | 5.6 |
| `ndws/next_day_wildfire_spread_eval_00.tfrecord` | 213,313,000 | 203.4 |
| `ndws/next_day_wildfire_spread_eval_01.tfrecord` | 187,075,501 | 178.4 |
| `ndws/next_day_wildfire_spread_test_00.tfrecord` | 213,313,000 | 203.4 |
| `ndws/next_day_wildfire_spread_test_01.tfrecord` | 146,972,657 | 140.2 |
| `ndws/next_day_wildfire_spread_train_00.tfrecord` | 213,313,000 | 203.4 |
| `ndws/next_day_wildfire_spread_train_01.tfrecord` | 213,313,000 | 203.4 |
| `ndws/next_day_wildfire_spread_train_02.tfrecord` | 213,313,000 | 203.4 |
| `ndws/next_day_wildfire_spread_train_03.tfrecord` | 213,313,000 | 203.4 |
| `ndws/next_day_wildfire_spread_train_04.tfrecord` | 213,313,000 | 203.4 |
| `ndws/next_day_wildfire_spread_train_05.tfrecord` | 213,313,000 | 203.4 |
| `ndws/next_day_wildfire_spread_train_06.tfrecord` | 213,313,000 | 203.4 |
| `ndws/next_day_wildfire_spread_train_07.tfrecord` | 213,313,000 | 203.4 |
| `ndws/next_day_wildfire_spread_train_08.tfrecord` | 213,313,000 | 203.4 |
| `ndws/next_day_wildfire_spread_train_09.tfrecord` | 213,313,000 | 203.4 |
| `ndws/next_day_wildfire_spread_train_10.tfrecord` | 213,313,000 | 203.4 |
| `ndws/next_day_wildfire_spread_train_11.tfrecord` | 213,313,000 | 203.4 |
| `ndws/next_day_wildfire_spread_train_12.tfrecord` | 213,313,000 | 203.4 |
| `ndws/next_day_wildfire_spread_train_13.tfrecord` | 213,313,000 | 203.4 |
| `ndws/next_day_wildfire_spread_train_14.tfrecord` | 208,833,427 | 199.2 |

## Dataset 1 — Kaggle smoke detection CSV

`kaggle_smoke/smoke_detection_iot.csv`, 5,834,376 bytes, sniffed as **UTF-8
text** (comma-delimited, single header row).

**62,630 data rows**, 16 columns:

| # | column | note |
|---|---|---|
| 0 | *(empty name)* | unnamed row index written by the exporter — a **third drop candidate** alongside CNT and UTC |
| 1 | `UTC` | **present, confirmed** — Unix epoch seconds (first row `1654733331`). To be dropped later. |
| 2 | `Temperature[C]` | |
| 3 | `Humidity[%]` | |
| 4 | `TVOC[ppb]` | vendor VOC index, not the raw gas resistance the planned node reports |
| 5 | `eCO2[ppm]` | |
| 6 | `Raw H2` | |
| 7 | `Raw Ethanol` | |
| 8 | `Pressure[hPa]` | |
| 9 | `PM1.0` | |
| 10 | `PM2.5` | |
| 11 | `NC0.5` | particle number concentration |
| 12 | `NC1.0` | |
| 13 | `NC2.5` | |
| 14 | `CNT` | **present, confirmed** — monotonic sample counter. To be dropped later. |
| 15 | `Fire Alarm` | label: 44,757 positive (`1`) / 17,873 negative (`0`) — 71.5 % positive |

**Identification of the drop columns is confirmed:** exact-match (case- and
whitespace-insensitive) lookups found `CNT` at index 14 and `UTC` at index 1.
Both are still in the file; this commit drops nothing.

Why they are drop candidates, for the record: `CNT` is a row counter that
rises monotonically through each recording session, and `UTC` is wall-clock
time, so either lets a model separate the alarm sessions from the ambient
sessions without looking at a sensor value at all. The unnamed index column at
position 0 has the same defect.

Two further observations worth carrying forward, neither acted on here:

- The label balance (71.5 % positive) is nothing like a real deployment, where
  positives are vanishingly rare. Any accuracy figure from this file is
  meaningless without re-balancing or a per-session split.
- The channel set does **not** match the planned Emberline node. This file has
  `TVOC[ppb]` and `eCO2[ppm]` (Bosch/vendor-derived indices) where the node's
  contract specifies raw `gas_ohms`; it has no equivalent of the rolling-
  baseline gas log-ratio in
  [docs/02_DATA_PIPELINE.md](../../docs/02_DATA_PIPELINE.md) §d.2. Treat it as
  a pretraining or sanity dataset, not as node data.

## Dataset 2 — NDWS (Next Day Wildfire Spread) tfrecords

`ndws/`, **19 files, 3,955,889,585 bytes (3.68 GiB)**.

| split | files | bytes |
|---|---|---|
| train | 15 | 3,195,215,427 |
| eval | 2 | 400,388,501 |
| test | 2 | 360,285,657 |

Format sniff: no ASCII magic number, as expected for TFRecord, which is a
length-prefixed record container (first 16 bytes of `train_00`:
`31 41 03 00 00 00 00 00 ac bf 53 2a 0a ad 82 0d` — an 8-byte little-endian
record length `0x0000000000034131` = 213,297, then its masked CRC32, then the
protobuf payload). Contents were **not** decoded: reading them needs
TensorFlow, which is not a dependency of this repository, and the task was
survey-only.

Note for later: 16 of the 19 shards are exactly 213,313,000 bytes — that is
the shard size, not a coincidence. The three that differ
(`train_14`, `eval_01`, `test_01`) are the last shard of each split. A real
manifest should record SHA-256 per file, since equal sizes make size-only
comparison useless.

## Not done here

No checksums, no `SOURCE.md`, no `MANIFEST.yaml`, no licence capture in
`external/`, no record counts for the tfrecords, no ingestion code. Those are
the next steps described in [README.md](README.md), and they are prerequisites
for any number from this data appearing anywhere.
