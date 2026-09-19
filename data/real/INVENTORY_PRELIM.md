# Preliminary real-data inventory

First scan **2026-09-18**; revised **2026-09-19** after the datasets were
moved into the repository and the team issued the authoritative constraints
now recorded in [README.md](README.md). Read-only throughout: nothing was
modified, renamed or ingested by this survey. This is a first look, not a
manifest — no checksums were computed and no tfrecord was decoded.

Method: `os.walk` over the raw tree; sizes from `stat()`; format sniffed from
the first 64 bytes (magic-number table, UTF-8 decode attempt as fallback). The
Kaggle CSV was read with the standard-library `csv` module to count rows and
list columns. The tfrecords were **not** opened beyond their first bytes.

## Where the data is (resolved 2026-09-19)

The first scan found both datasets *outside* the repository, one directory
above it. They have since been **moved into the repo** at
`emberline/data/real/raw/`, which was option 1 of the two the first scan
offered. Repo-relative paths therefore now work:

| dataset | path (relative to repo root) |
|---|---|
| Kaggle smoke CSV | `data/real/raw/kaggle_smoke/` |
| NDWS tfrecords | `data/real/raw/ndws/` |

Re-verified with the real files present: 3.7 GiB on disk,
`git status` clean of it, `git add -A` stages none of it,
`git check-ignore -v` attributes both dataset trees to the
`data/real/raw/**` rule. The bulk is safely out of git.

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

### Three leak columns, and how badly they leak (measured 2026-09-19)

All three are confirmed present, and **all three must be dropped**; a feature
builder should assert their absence rather than trust a comment.

| # | column | what it is | leak strength, measured |
|---|---|---|---|
| 0 | *(empty name)* | exactly `0..62629`, the exporter's row index (verified) | a single threshold on it alone scores **76.4 %** accuracy |
| 14 | `CNT` | sample counter, 0–24,993, resetting **4 times** (so ~5 recording sessions) | a single threshold alone scores **90.0 %** accuracy; no negative row has `CNT > 5,743`, so that one cut labels 38,500 rows — 86 % of all positives — perfectly |
| 1 | `UTC` | epoch seconds, span 116.1 h, **not** monotonic across the file | separates sessions by wall-clock time |

For scale: the majority-class baseline is 71.5 %, and the synthetic detector
reported in [REPORT.md](../../REPORT.md) reaches F1 0.952. A model left with
`CNT` can reach 90 % accuracy while reading no sensor value at all, so any
result computed before these columns are dropped is meaningless.

The `CNT` resets are useful, though: they mark the session boundaries, which
is what a session-level split needs, since the contract in
[docs/02_DATA_PIPELINE.md](../../docs/02_DATA_PIPELINE.md) §d.3 splits by
session, never by window. Derive the split from `CNT` resets, then drop `CNT`.

### The gas channel is a vendor index, and it runs backwards

The node contract specifies raw `gas_ohms`. This file has `TVOC[ppb]` and
`eCO2[ppm]`, which are Bosch/vendor-derived indices. Per the authoritative
constraints the log-ratio-to-rolling-baseline feature is computed on `TVOC`
here, and two measured facts must be handled when it is:

- **4.3 % of TVOC samples are exactly zero** (2,698 of 62,630; min 0, median
  981, max 60,000). `log(0)` is undefined and a rolling baseline can itself be
  zero, so the feature needs an epsilon and a defined behaviour when the
  baseline is degenerate. Zeros are not confined to one class (2,038 negative,
  660 positive), so they cannot simply be dropped.
- **The direction is inverted relative to the node's physics.** Raw gas
  resistance *falls* in smoke, so on a node `log(gas / baseline)` goes
  negative during an event. In this dataset TVOC is *higher* when there is no
  alarm: mean **4,596.6 ppb for `Fire Alarm = 0`** against **882.0 ppb for
  `Fire Alarm = 1`**. Whatever this label marks, it is not "VOC went up". A
  feature ported from the node formula without checking sign will point the
  wrong way, and a model trained here will learn a relationship that does not
  transfer.

This is the domain gap in concrete terms, and it is the argument for our own
logged sessions: this dataset can pretrain or sanity-check a pipeline, but it
cannot stand in for node data.

### Reporting rule for anything derived from this file

Label balance is 71.5 % positive (44,757 / 17,873), nothing like deployment,
where positives are vanishingly rare. Report **PR-AUC and false positives per
node-day**; accuracy is uninformative at this prior, and the prior mismatch
must be stated next to any number quoted from this dataset.

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

**Manifests must use SHA-256, never size.** 16 of the 19 shards are exactly
213,313,000 bytes — that is the shard size, not a coincidence; the three that
differ (`train_14`, `eval_01`, `test_01`) are the last shard of each split.
Size comparison therefore cannot detect a swapped, truncated or re-downloaded
file, so `MANIFEST.yaml` identifies every file by SHA-256.

**Decoding needs TensorFlow, which this repository does not depend on.**
Either add `tensorflow-cpu` and record it as a Phase-3 dependency in
`pyproject.toml`, or use a lightweight TFRecord parser if one proves reliable
on these files. Either way the choice is recorded, not made silently: adding a
heavyweight dependency changes what a fresh clone must install to run
`bash verify.sh`.

## Not done here

No checksums, no `SOURCE.md`, no `MANIFEST.yaml`, no licence capture in
`external/`, no record counts for the tfrecords, no ingestion code. Those are
the next steps described in [README.md](README.md), and they are prerequisites
for any number from this data appearing anywhere.
