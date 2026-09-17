# Emberline MVP run summary

Generated 2026-09-17T00:38:02+00:00 by `python -m emberline.mvp` (started 2026-09-17T00:35:56+00:00, 126 s wall).

> Every value on this page was read from an artifact this run produced or from `metrics/*.json`; none was typed in. **Everything is synthetic simulation** - see [03_RESULTS.md](03_RESULTS.md) and [04_GAP_REGISTER.md](04_GAP_REGISTER.md).

## Environment

| item | value |
|---|---|
| commit | `d7a2257` on `emberline-build` |
| platform | Windows-11-10.0.26200-SP0 |
| python | 3.13.14 |
| torch | 2.6.0+cpu |
| cpu count | 12 |

## Steps

| step | command | result | wall |
|---|---|---|---|
| verify | `bash.exe verify.sh` | OK | 71 s |
| demo | `python.exe -m emberline.demo --scenario ridgeline --fast --wind-shift 40 --kill-node N3` | OK | 49 s |
| pitch assets | `python.exe -m emberline.demo.pitch_assets` | OK | 5 s |

## Test suite

**PASS** - 61 passed, 0 failed, 0 errors, 0 skipped; `VERIFY: ALL GREEN` printed.

## Scoreboard of this run (`metrics/demo_last_run.json`)

| quantity | value |
|---|---|
| scenario / forecast backend | ridgeline / surrogate |
| ignition → Tier-0 chirp | 30 s |
| ignition → Tier-1 voice + bearing | 120 s |
| ignition → Tier-2 full cascade | 300 s |
| ignition estimate error | 248 m |
| cascade max hops / median alert latency | 1 / 0.4 s |
| node killed / at / self-heal announced at | N3 / t+390 s / t+570 s |
| wind shift / at | +40° / t+870 s |
| access points moved to another exit by the re-plan | 0 |
| forecast wall-clock per ensemble | [16.65, 16.47] s |
| mesh channel utilisation | 0.21% |
| mesh transmissions / collisions | 851 / 0 |
| truth burned area at end | 145.0 ha |

## Standing metrics the scoreboard read (`metrics/surrogate.json`, `metrics/detect.json`)

| quantity | value |
|---|---|
| surrogate IoU +10/+30/+60, val worlds | 0.678 / 0.627 / 0.507 |
| surrogate IoU@+30, held-out wind regime | 0.657 |
| worst-case p5 IoU@+30 | 0.369 |
| arrival MAE (val) | 3.7 min |
| ensemble speedup vs physics | 0.5× |
| detector F1, CNN / GBM | 0.943 / 0.952 |
| false positives per node-day (CNN) | 7.33 |
| detection latency median | 36 s |

Cross-check that REPORT.md still quotes these values: val IoU@+30 0.627 PASS, held-out IoU@+30 0.657 PASS, p5 IoU@+30 0.369 PASS, speedup 0.5x PASS, CNN F1 0.943 PASS, GBM F1 0.952 PASS, FP/node-day (CNN) 7.33 PASS, ECE raw 0.060 PASS.

### Demo scoreboard as printed

```text
  Surrogate IoU +10/+30/+60 (val worlds)    0.678 / 0.627 / 0.507
    held-out wind regime IoU@+30            0.657
    worst-case (p5) IoU@+30                 0.369
    fire-arrival MAE                        3.7 min
    ensemble speedup vs physics             0.5×
  Detector F1 (CNN vs GBM, val events)      0.943 vs 0.952
    ambient false positives                 7.3 /node-day
    detection latency (median)              36 s
  This run: ignition → Tier-0               30 s
  This run: ignition → Tier-1               120 s
  This run: ignition → Tier-2 cascade       300 s
  This run: ignition estimate error         248 m
  This run: mesh channel utilization        0.21 %
  This run: truth burned area at t=75 min   145 ha
```

## Artifacts

| artifact | path | present |
|---|---|---|
| demo GIF | `demo/out/demo.gif` | yes |
| demo frames | `demo/out/demo_frames/` (15 PNGs) | yes |
| raw run arrays / plans | `demo/out/last_run_artifacts.npz`, `demo/out/last_run_artifacts.json` | yes |
| run metrics | `metrics/demo_last_run.json` | yes |
| pitch asset `system_architecture` | `demo/out/pitch_assets/system_architecture.png` | yes |
| pitch asset `cone_evacuation_before_after` | `demo/out/pitch_assets/cone_evacuation_before_after.png` | yes |
| pitch asset `mesh_topology` | `demo/out/pitch_assets/mesh_topology.png` | yes |
| pitch asset `detection_confusion` | `demo/out/pitch_assets/detection_confusion.png` | yes |
| pitch asset `calibration_before_after` | `demo/out/pitch_assets/calibration_before_after.png` | yes |
| pitch asset `warning_timeline` | `demo/out/pitch_assets/warning_timeline.png` | yes |
| CAP draft (newest in outbox/) | `outbox/cap_1_tier2_2026-09-17T003730+0000.json` | yes, status Exercise |

The CAP draft is a JSON document with `status: "Exercise"`; nothing transmits it. Frames, GIF and the outbox are gitignored; the six pitch PNGs are committed.

---

Synthetic simulation only. No physical node, radio, real sensor data, real terrain or real fire record was involved in producing any number above. Index: [README.md](../README.md).
