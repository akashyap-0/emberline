# `data/real/` — the real-data workspace

Everything under this directory is **real-world data**, as opposed to the
synthetic worlds the rest of the repository generates from `seed: 1337`. The
two must never be mixed: real results are reported separately
(`REPORT_REAL.md`, planned), never merged into the synthetic tables in
[REPORT.md](../../REPORT.md). See
[docs/02_DATA_PIPELINE.md](../../docs/02_DATA_PIPELINE.md) for the boundary
and [docs/04_GAP_REGISTER.md](../../docs/04_GAP_REGISTER.md) for what is still
missing.

This directory is ingested exclusively through `emberline/data/`
(`python -m emberline.data.ingest`, `python -m emberline.data.ndws`), whose
loaders refuse any path under a `data/synthetic` directory or elsewhere in
the synthetic stack's `data/` tree (`emberline.data.assert_real_data_path`;
tests prove the guard fires). [INVENTORY.md](INVENTORY.md) is the
authoritative content listing with SHA-256 digests;
[MANIFEST.yaml](MANIFEST.yaml) is its machine-readable form plus the split
assignment of every training group.

## The directory contract

| directory | contents | mutability | in git? |
|---|---|---|---|
| `raw/` | datasets **exactly as downloaded**. Original filenames, original bytes, original archive layout. | **Read-only.** Never edit, rename, re-encode, de-duplicate or "clean" a file in place. If something is wrong with a raw file, fix it downstream in `interim/`, not here. | no (bulk ignored); `SOURCE.md` / `MANIFEST.yaml` / `INVENTORY.md` yes |
| `external/` | the paperwork *about* the data: dataset cards, licence texts, README copies from the source, paper PDFs, saved web pages, e-mail permissions, burn-site notes. | Append-only in practice. | **yes, fully tracked** (keep it small; link rather than mirror large PDFs) |
| `interim/` | normalised, uninterpreted intermediates — one row per sample, consistent dtypes, UTC timestamps, **Parquet**. Decoding, unit fixes and schema alignment happen here. No feature engineering, no filtering for a model. | Regenerable. Safe to delete and rebuild from `raw/`. | no (bulk ignored) |
| `processed/` | model-ready sets: windowed, feature-extracted, split-assigned, Parquet. Whatever a training or evaluation script consumes directly. | Regenerable. Safe to delete and rebuild from `interim/`. | no (bulk ignored) |

Flow is one-directional and each hop is reproducible:

```
raw/  ──decode+normalise──▶  interim/  ──window+features+split──▶  processed/  ──▶  model
 ▲ read-only                  ▲ regenerable                         ▲ regenerable
 └── external/ documents where raw/ came from and under what licence
```

The rule that matters: **you must be able to delete `interim/` and
`processed/` entirely and rebuild them from `raw/` with a command.** If that
is not true, something belongs in `raw/` that is not there.

## What git tracks

Bulk data never enters git. The following *are* tracked, so the contract and
the provenance survive a fresh clone:

- `.gitkeep` in each directory (keeps the skeleton),
- `SOURCE.md` — where one dataset came from (see below),
- `MANIFEST.yaml` — per-file checksums and counts for one dataset,
- `INVENTORY.md` (and this round's `INVENTORY_PRELIM.md`) — what is on disk,
- everything in `external/`.

The rules are at the bottom of [.gitignore](../../.gitignore). Verify a path
before assuming: `git check-ignore -v data/real/raw/<something>`.

## Adding a new dataset — the 4-step recipe

The pipeline is built so a better dataset later (our own ESP32 logger
sessions, the supervised burn) drops in without redesign:

1. **Drop the files** in one new directory per dataset under `raw/`,
   `lower_snake_case`, exactly as downloaded — no repacking, no cleaning.
   Save the dataset card / licence / paper into `external/` at the same
   time, and write `raw/<dataset>/SOURCE.md` (URL, exact version, who/when,
   licence, one paragraph on what it is and why Emberline wants it).
2. **Declare the schema** in `emberline/data/schemas.py`: the exact columns
   or feature spec as they truly are (leak columns and all — dropping is a
   downstream modelling decision). The validator must reject unknown and
   missing columns, never coerce. The ESP32 logger contract
   (`ms,pm1,pm25,pm10,gas_ohms,temp_c,rh,press_hpa`) is already declared and
   waiting for its first data.
3. **Teach `emberline/data/ingest.py` the dataset**: validate against the
   schema, normalize to parquet/npy in `interim/` (one file per recording
   session or shard), and record it in `MANIFEST.yaml` — path, SHA-256,
   bytes, source id, licence note, and the SPLIT assignment of every
   training group, assigned once and deterministically (by session/shard,
   never randomly by row). Re-running must be idempotent.
4. **Add the tests and re-inventory**: schema rejection, manifest
   determinism, split stability (mirror `tests/test_real_data_infra.py`),
   then refresh `INVENTORY.md` (size + SHA-256 per file; identity is the
   digest, never the size) and run `bash verify.sh`.

Large data is deliberately **not** stored in this repository and not in git
LFS. A teammate reproduces a result by re-downloading from `SOURCE.md` and
checking `MANIFEST.yaml`, not by cloning gigabytes.

## Authoritative constraints

Team decisions of 2026-09-19. These bind any ingestion, feature or reporting
work that touches this directory; they are not suggestions.

1. **`make` is unavailable on the team machine.** Use `bash verify.sh`
   wherever a prompt or doc says `make verify`. (`make mvp` →
   `python -m emberline.mvp`, which already falls back to bash.)
2. **The Kaggle CSV has three leak columns, not two.** Drop the unnamed
   index column at position 0 as well as `CNT` and `UTC`. Any feature
   builder must assert that none of the three survives into the feature
   matrix — a test, not a comment.
3. **The gas channel is a vendor index, not raw resistance.** The
   log-ratio-to-rolling-baseline feature is computed on `TVOC[ppb]` for this
   dataset. The node contract specifies raw `gas_ohms`. This domain gap must
   be stated explicitly in the report; it is the reason our own logged
   sessions are required rather than optional. Two consequences measured in
   [INVENTORY_PRELIM.md](INVENTORY_PRELIM.md) and carried into any
   implementation: the log-ratio needs an epsilon because 4.3 % of TVOC
   samples are exactly zero, and TVOC in this dataset runs **opposite** to
   the direction the node's physics implies.
4. **Report PR-AUC and false positives per node-day, not accuracy.** The
   Kaggle label balance is 71.5 % positive, nothing like deployment, where
   positives are vanishingly rare. State the prior mismatch in the report
   next to any number derived from this dataset.
5. **Manifests identify files by SHA-256, never by size.** Sixteen of the
   nineteen NDWS shards are byte-identical in size, so size comparison
   cannot detect a swapped or truncated file.
6. **Decoding NDWS requires TensorFlow**, which is not a dependency of this
   repository. Either add `tensorflow-cpu` and record it as a Phase-3
   dependency in `pyproject.toml`, or use a lightweight TFRecord parser if
   one proves reliable. Do not add a heavy dependency silently.
   *Resolved 2026-09-18:* `emberline/data/tfrecord_lite.py`, a pure-Python
   TFRecord + `tf.train.Example` reader, parses all 19 shards (18,545
   records) and is covered by round-trip tests; TensorFlow was NOT added.

## Location and ignore rules (verified 2026-09-19)

The datasets now live inside the repository at `data/real/raw/kaggle_smoke/`
and `data/real/raw/ndws/` (they were previously a level above it, outside
git). The ignore rules were re-checked **with the real files present**:
3.7 GiB sits on disk, `git status` reports none of it, and `git add -A`
stages none of it. Confirm any new path before assuming:

```bash
git check-ignore -v data/real/raw/<something>
```
