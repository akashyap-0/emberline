# `data/real/` — the real-data workspace

Everything under this directory is **real-world data**, as opposed to the
synthetic worlds the rest of the repository generates from `seed: 1337`. The
two must never be mixed: real results are reported separately
(`REPORT_REAL.md`, planned), never merged into the synthetic tables in
[REPORT.md](../../REPORT.md). See
[docs/02_DATA_PIPELINE.md](../../docs/02_DATA_PIPELINE.md) for the boundary
and [docs/04_GAP_REGISTER.md](../../docs/04_GAP_REGISTER.md) for what is still
missing.

**Nothing in this directory is ingested by any code yet.** As of this commit
it is a skeleton plus an inventory; no loader, no feature extractor, no
training path reads from here. See
[INVENTORY_PRELIM.md](INVENTORY_PRELIM.md) for what is on disk today.

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

## Where to drop a new dataset

1. Create **one directory per dataset** under `raw/`, named in
   `lower_snake_case` after the source, e.g. `raw/kaggle_smoke/`,
   `raw/ndws/`. Do not nest datasets or mix two sources in one folder.
2. Put the files in **exactly as downloaded**. Keep the archive too if it is
   small; do not repack.
3. Write `raw/<dataset>/SOURCE.md` before you do anything else with it:
   - the URL and the exact dataset version/date you downloaded,
   - who downloaded it, when (UTC), and with what tool,
   - the licence and what it permits (redistribution? commercial use?),
   - the citation, if the source asks for one,
   - one paragraph: what the data actually is, and what Emberline wants it for.
4. Write `raw/<dataset>/MANIFEST.yaml`: for each file, its size in bytes and
   its SHA-256, plus the row/record count if cheap to obtain. This is how a
   later run proves the bytes did not change.
5. Save the dataset card, licence and any paper into `external/`.
6. Re-run the inventory and update `INVENTORY.md`.

Large data is deliberately **not** stored in this repository and not in git
LFS. A teammate reproduces a result by re-downloading from `SOURCE.md` and
checking `MANIFEST.yaml`, not by cloning gigabytes.

## Current location caveat (2026-09-18)

The Kaggle smoke CSV and the NDWS tfrecords that have been downloaded are
**not** in this directory. They sit one level up, outside the git repository,
at `conrad_challenge_fire_prototype/data/real/raw/`. That path is a sibling of
the repo, not part of it. Nothing has been moved — see
[INVENTORY_PRELIM.md](INVENTORY_PRELIM.md) for the exact paths and the two
options for resolving it.
