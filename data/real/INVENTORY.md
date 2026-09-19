# Real-data inventory

Authoritative inventory of everything under `data/real/raw/`, computed
**2026-09-18** (local). Supersedes
[INVENTORY_PRELIM.md](INVENTORY_PRELIM.md), which was a survey without
checksums; the prelim's measured findings about the smoke CSV (leak columns,
TVOC polarity, zeros) remain valid and are referenced from here rather than
repeated.

Method: every regular file under `data/real/raw/` (excluding `.gitkeep`);
size from `stat()`, digest via `sha256sum` over the full file bytes.
Nothing was modified, renamed, or decoded by this pass.

**Identity is SHA-256, never size.** 16 of the 19 NDWS shards are
byte-identical in size (213,313,000 bytes — the fixed shard size), so size
cannot distinguish a swapped, truncated-and-repadded, or re-downloaded shard.
Any integrity check against this inventory must compare digests.

## Summary

| | files | bytes | size |
|---|---|---|---|
| all | 20 | 3,961,723,961 | 3.69 GiB |
| `kaggle_smoke/` | 1 | 5,834,376 | 5.6 MiB |
| `ndws/` | 19 | 3,955,889,585 | 3.68 GiB |

## Kaggle smoke detection CSV

Source: Kaggle "Smoke Detection Dataset" (`deepcontractor/smoke-detection-dataset`),
based on Stefan Blattmann's real IoT smoke-detector recordings.
Licence: **not yet captured** — the Kaggle card must be saved into
`external/` and the licence confirmed there before any redistribution;
internal training/evaluation use only until then.
Format: UTF-8 CSV, one header row, 62,630 data rows, 16 columns
(unnamed row index, `UTC`, 12 sensor channels, `CNT`, `Fire Alarm` label).

| file | bytes | format | SHA-256 |
|---|---|---|---|
| `kaggle_smoke/smoke_detection_iot.csv` | 5,834,376 | CSV (UTF-8) | `7bbf5deaab4c94746b344cee7bfec493f69b7d624ea63f35fabcd3c4a4e6eb7d` |

## NDWS (Next Day Wildfire Spread) tfrecords

Source: "Next Day Wildfire Spread" (Huot et al., 2022,
arXiv:2112.02447; Kaggle `fantineh/next-day-wildfire-spread`). The paper
and dataset card describe it as openly released (commonly cited as
CC BY 4.0); licence text **not yet captured** into `external/` — confirm
there before any redistribution.
Format: TFRecord (length-prefixed protobuf `tf.train.Example` records),
1 km resolution. The dataset ships its own split — 15 train / 2 eval /
2 test shards — which is respected as-is downstream (no reshuffling).

| file | bytes | format | SHA-256 |
|---|---|---|---|
| `ndws/next_day_wildfire_spread_eval_00.tfrecord` | 213,313,000 | TFRecord | `85522369187adc7e0f76f7cb9e45ad55c47cc8351304ce726bee27de20c32d7b` |
| `ndws/next_day_wildfire_spread_eval_01.tfrecord` | 187,075,501 | TFRecord | `a7b3b5b8915072cb3e1a5b4a82091f97202ca48d69db4546565620e5c52a9d8f` |
| `ndws/next_day_wildfire_spread_test_00.tfrecord` | 213,313,000 | TFRecord | `d2e156708b478206f55816447825c076479dc7ee1a7b940991ac2577f10f2bf0` |
| `ndws/next_day_wildfire_spread_test_01.tfrecord` | 146,972,657 | TFRecord | `4ae153931a3ca642e850871713e766c760c91ae1148749004e378da84a93603b` |
| `ndws/next_day_wildfire_spread_train_00.tfrecord` | 213,313,000 | TFRecord | `c67274eb55c44511188eceddf4915c579985025db3fb4e9558459167c11ebaa8` |
| `ndws/next_day_wildfire_spread_train_01.tfrecord` | 213,313,000 | TFRecord | `3e75e5411d78ffe44a9cafd752e63f0a8adc7d09edca21b191ac2a14fe45cd73` |
| `ndws/next_day_wildfire_spread_train_02.tfrecord` | 213,313,000 | TFRecord | `0e52dd94f56acd5ce75be9a5cf7e8ee874776e3906f8c5e610e475f291456f6b` |
| `ndws/next_day_wildfire_spread_train_03.tfrecord` | 213,313,000 | TFRecord | `b26bd4a23e76077a962bc952367d87ef632a762df5a5e40873d73adab24bdf72` |
| `ndws/next_day_wildfire_spread_train_04.tfrecord` | 213,313,000 | TFRecord | `95919adcdcd51dba0fc311a17a93355c2056662040f4123c2fdaa15b31a1cb55` |
| `ndws/next_day_wildfire_spread_train_05.tfrecord` | 213,313,000 | TFRecord | `f20e63bfe62e0a4b5ae6533a62762676fc83ae2a657c52dfa6d4299e2613a9a0` |
| `ndws/next_day_wildfire_spread_train_06.tfrecord` | 213,313,000 | TFRecord | `fb35480aac3612978bc0134ff6fe6f60fe31d3b6425df307409ecf3bd2e35cf5` |
| `ndws/next_day_wildfire_spread_train_07.tfrecord` | 213,313,000 | TFRecord | `545d06e898f0622bd9352f31857544a86ed15fc40446dd00d5ca1d78425383a5` |
| `ndws/next_day_wildfire_spread_train_08.tfrecord` | 213,313,000 | TFRecord | `f362386c22aeb4b043c520c995a64954874e6dfa8a8af10d67e1feeb9e3014dd` |
| `ndws/next_day_wildfire_spread_train_09.tfrecord` | 213,313,000 | TFRecord | `b041e7786b5e49a46fb5784ebc7cc8dcfc96b4f12b12924e1a5420a2647415cc` |
| `ndws/next_day_wildfire_spread_train_10.tfrecord` | 213,313,000 | TFRecord | `8ae39ce20888f2f306cdd512dd3f278eba8a0a17a036c62ac2a83f23cc709eb1` |
| `ndws/next_day_wildfire_spread_train_11.tfrecord` | 213,313,000 | TFRecord | `c624d953a7f42ed70ee845e5fa073b0d1111b5b81863b7b480a1f34b0a1b9f71` |
| `ndws/next_day_wildfire_spread_train_12.tfrecord` | 213,313,000 | TFRecord | `8f1891152c24d5fe57c76a48b93d5c232441273119e45a9ab2b19da061528dcd` |
| `ndws/next_day_wildfire_spread_train_13.tfrecord` | 213,313,000 | TFRecord | `d39a4bd03196df62c2462a53b0415392631da00f53076ccd90e24a9efeef76c9` |
| `ndws/next_day_wildfire_spread_train_14.tfrecord` | 208,833,427 | TFRecord | `e8712ec961ca15295698be4daa60285ee6a80cccdc37165097a9636c7be50d5f` |

## Verifying

```bash
cd data/real
sha256sum -c <(awk -F'|' '{print $3"  ../../"$1}' <inventory-as-pipe-list>)
```

or simply re-run the hash over `raw/` and diff against the tables above.
`data/real/MANIFEST.yaml` (written by `python -m emberline.data.ingest`)
carries the same digests in machine-readable form, plus split assignments.
