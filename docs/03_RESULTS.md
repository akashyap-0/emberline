# Emberline measured results — summary of REPORT.md

A faithful condensation of [REPORT.md](../REPORT.md). Every number below was
copied from that file or from the `metrics/*.json` it was rendered from; none
was recomputed, rounded up or "expected". Where a target was missed, the miss
is stated. Read REPORT.md's *Limitations* section before quoting anything.

**All results are on synthetic, procedurally generated worlds.** They are
evidence that the software pipeline works and of how it behaves under its
own models. They are not evidence about real-world detection, forecasting or
warning performance.

Regeneration on a fresh clone requires the gitignored datasets first:
`python -m emberline.surrogate.data` (tens of minutes) and
`python -m emberline.detect.data`. The demo, tests and hindcasts run from the
committed checkpoints alone.

## Headline table

| metric | value | target | status | regenerate |
|---|---|---|---|---|
| Surrogate IoU@+30, val worlds | 0.627 | 0.80 | **missed** | `python -m emberline.surrogate.eval` |
| Surrogate IoU@+30, held-out wind regime | 0.657 | — | regime gap closed (−0.030) | same |
| Surrogate worst-case p5 IoU@+30 (val) | 0.369 | — | — | same |
| Fire-arrival MAE, val / held-out | 3.7 / 3.0 min | — | — | same |
| Ensemble speedup vs repo physics (20 members × 60 min) | 0.5× | 100× | **missed** (see gap) | same |
| Ensemble calibration ECE, raw, val fires (shipped) | 0.060 | — | fitted T not shipped | `python -m emberline.surrogate.calibration` |
| Detection F1, CNN / GBM (val events) | 0.943 / 0.952 | beat baseline | **CNN did not beat GBM** | `python -m emberline.detect.eval` |
| False positives per node-day, ambient day, CNN / GBM | 7.33 / 8.67 | — | single-node figure; mesh corroboration mitigates | same |
| Detection latency, median / mean | 36 s / 86 s | — | 3 of 25 fires undetected | same |
| Canonical demo: Tier-0 / Tier-1 / Tier-2 after ignition | 30 / 120 / 300 s | — | — | `python -m emberline.demo --scenario ridgeline --fast --kill-node N3 --wind-shift 40` |
| Hindcast warning minutes gained (6 authored scenarios) | +19.5, +7.5, −6.5, three misses | — | mixed, documented | `python -m emberline.foresight.hindcast scenarios/*.yaml` |
| Stress tests | 6/6 pass; 2 designed limitations pinned | — | — | `python -m pytest tests/test_stress.py -v` |
| Test suite | 54 tests pass | — | — | `bash verify.sh` |

## 1. Neural surrogate (Phase 3, retrained Phase 9)

Splits are by world; the held-out wind regime (80–130°) never appeared in
training. 64 fires per split. Checkpoint `data/checkpoints/best.pt`.

| split | fires | IoU@+10 | IoU@+30 | IoU@+60 | arrival MAE (min) |
|---|---|---|---|---|---|
| val (unseen worlds) | 64 | 0.678 | 0.627 | 0.507 | 3.7 |
| held-out wind regime | 64 | 0.651 | 0.657 | 0.523 | 3.0 |

Worst case: 5th-percentile IoU@+30 across val fires = 0.369.

Ensemble wall-clock, 20 members × 60 sim-min, 4 CPU threads: physics 6.82 s
vs surrogate 14.23 s → **0.5×**.

**Gap analysis (from REPORT.md, in spirit):**

- IoU@+30 = 0.627 misses the 0.80 aspirational target. Main error mode is
  autoregressive drift: small front-position errors compound over three
  steps. More training worlds and longer training (the run was CPU-budget
  capped) are the obvious levers.
- Speedup 0.5× misses the 100× target. The physics baseline is the repo's own
  heavily vectorised cellular automaton (6.8 s for 20 members × 60 min), not
  an operational solver, so the denominator is unusually fast. Against
  FARSITE-class physics the batched one-forward-per-10-min rollout would be
  faster by orders of magnitude; against this CA it is slower. The claim
  "surrogate is faster than physics" is **not supported** by this repo's
  measurement and must not be made.

### Wind-augmentation retraining (Phase 9): before / after

v1 = `best_v1.pt` (archived, pre-augmentation). v2 = `best.pt` (fine-tuned
from v1 with joint world+wind rotation). Same machine, same 64 fires per
split, same eval code.

| metric (val / held-out wind regime) | v1 pre-augmentation | v2 wind-augmented |
|---|---|---|
| IoU@+10 | 0.551 / 0.296 | 0.678 / 0.651 |
| IoU@+30 | 0.655 / 0.411 | 0.627 / 0.657 |
| IoU@+60 | 0.621 / 0.478 | 0.507 / 0.523 |
| arrival MAE (min) | 4.244 / 9.406 | 3.674 / 2.986 |
| worst-case p5 IoU@+30 (val) | 0.248 | 0.369 |

Held-out-regime IoU@+30 moved +0.246; the val–holdout gap went from 0.244 to
−0.030 (closed). Costs: val IoU@+30 moved −0.028 and val IoU@+60 moved
−0.115, because capacity now spreads across all wind orientations and the
six-step +60 rollout pays most. The trade was accepted because detection-time
routing uses the +10/+30 cones.

Provenance note: the original laptop build quoted v1 at val 0.681 /
held-out 0.368. Re-measured from the same archived checkpoint on the Round-2
machine, v1 scored 0.655 / 0.411. Torch CPU kernel differences compound over
autoregressive steps; both records are preserved in REPORT.md. Regenerate the
comparison with `python -m emberline.surrogate.eval --compare-baseline metrics/surrogate_v1.json`.

## 2. Ensemble calibration + temperature scaling (Phase 10)

20-member ensemble, P(burn by +30 min) vs physics on 12 val-world fires,
near-fire cells only. The raw curve is under-confident (observed frequencies
exceed predictions).

| configuration | ECE on the val fires |
|---|---|
| raw ensemble — **what ships** (`foresight.temperature: null`) | **0.060** |
| with the fitted T = 2.37 (not shipped) | 0.196 |

Fit-then-verify protocol: T fitted on 12 fires from fresh worlds 161–168,
checked on 6 fires from disjoint worlds 169–172, where it improved ECE 0.290 →
0.267 (check passed). It then **failed to transfer** to the val population
(0.060 → 0.196), so no scaling ships. Caveats stated in REPORT.md: each
population estimate rests on 6–12 fires, and the val split also selected the
training checkpoint. Two naive fit objectives failed measurably first (plain
NLL: T = 2.59, val ECE 0.060 → 0.229; direct binned-ECE minimisation: T ran
to the grid edge, val ECE 0.330) and are documented in `fit_temperature`.

Baselines: pre-Phase-9 REPORT quoted ECE 0.113 (raw); re-measured here from
`best_v1.pt`, raw 0.148. Most of the calibration improvement came from the
Phase-9 retraining itself: 0.113 (v1, quoted) → 0.060 (v2 raw). Curves:
`demo/out/calibration_v2.png`.

## 3. Smoke detection (Phase 4)

Split by event; thresholds CNN 0.60 (train-selected), GBM 0.5; 1,581 val
windows.

| model | precision | recall | F1 |
|---|---|---|---|
| 1D-CNN (9,129 params) | 0.934 | 0.953 | 0.943 |
| GBM baseline | 0.953 | 0.952 | 0.952 |

The CNN does **not** beat the GBM on val F1. REPORT.md's engineering call:
with 60 s of context the handcrafted features capture most of the signal, so
ship the cheaper model on-node and revisit with longer windows.

Per-confounder false-positive rate on val windows:

| confounder | windows | CNN FP rate | GBM FP rate |
|---|---|---|---|
| bbq | 96 | 0.125 | 0.083 |
| wood_stove | 64 | 0.312 | 0.188 |
| vehicle | 28 | 0.107 | 0.107 |
| fog | 192 | 0.000 | 0.000 |
| dust | 136 | 0.000 | 0.000 |
| aerosol | 9 | 0.000 | 0.111 |
| ambient | 540 | 0.000 | 0.000 |

Wood stoves are the hardest confounder (they *are* wood smoke); fog and dust
are fully separated by their RH signature. The aerosol row rests on 9
windows.

Ambient day: 24 h, fire-free, 14 confounder events, 12 nodes, 34,548
windows → **7.33 false positives per node-day (CNN)**, 8.67 (GBM). These are
single-node, single-window numbers; the mesh corroboration ladder is what
turns them into (or refuses) siren-worthy alarms.

Detection latency over 25 fresh simulated fires: median **36 s**, mean 86 s
from plume arrival to the first positive window; 3 fires never produced a
classifiable plume at any node (burning away from the network).

## 4. Mesh behaviour

REPORT.md has no standalone mesh table; mesh numbers come from the canonical
demo run and from bounds pinned by tests.

| quantity | value | source |
|---|---|---|
| demo: ignition → Tier-0 / Tier-1 / Tier-2 | 30 s / 120 s / 300 s | `metrics/demo_last_run.json` |
| demo: cascade median per-alert latency after escalation | 0.4 s | same |
| demo: cascade max hops | 1 (all 12 nodes are mutual neighbours at n = 2.9) | same, `demo/out/last_run_artifacts.json` |
| demo: channel utilisation | 0.21 % | same |
| demo: transmissions / collisions over 75 min | 851 / 0 | same |
| demo: N3 killed → mourned (self-heal log line) | t+390 s → t+570 s | same |
| demo: ignition estimate error from bearings | 248 m | same |
| demo: road edges cut by the routing cone, before / after +40° shift | 6 / 0 | `last_run_artifacts.json` |
| pinned bound: 6-hop flood latency (dense-canopy exponent 4.5) | < 30 s | `tests/test_mesh.py::test_propagation_latency_six_hops` |
| pinned bound: 50 simultaneous triggers | < 1,250 transmissions, utilisation < 5 %, still reaches Tier 2 | `test_storm_suppression_50_triggers` |
| pinned bound: duty cycle | ≤ 1 % airtime per node over 1 h | `test_duty_cycle_budget_respected` |

These are outputs of the radio **model** (log-distance + occlusion
heuristic, single collision domain, no fading). No radio has been measured.

## 5. Hindcast harness (Phase 11, 6 authored scenarios)

Minutes from ignition. Scenario files are hand-authored patterns, not fire
records.

| scenario | conditions | Tier-0 | Tier-2 cascade | first 911 report (authored) | warning minutes gained |
|---|---|---|---|---|---|
| dry_ridge_evening | 7 m/s, nearest node 152 m | 0.0 | 2.5 | 22.0 | **19.5** |
| valley_night | 4 m/s, nearest node 212 m | 1.0 | — | 55.0 | **—** |
| highwind_ridge_run | 14 m/s, nearest node 786 m | 4.5 | 7.5 | 15.0 | **7.5** |
| stagnant_far_corner | 3 m/s, nearest node 1052 m | — | — | 65.0 | **—** |
| town_origin_fire | 6 m/s, nearest node 453 m | 5.0 | 12.5 | 6.0 | **−6.5** |
| degraded_mesh_ridge | 7 m/s, 2 nodes down (N0, N7), nearest node 396 m | 5.0 | — | 22.0 | **—** |

"—" means the mesh never corroborated to a cascade. Measured outcomes:
cascades in 3 of 6; misses at valley_night (Tier-0 only), stagnant_far_corner
(zero detections in 80 min), degraded_mesh_ridge (Tier-0 only with the two
best-placed nodes dead).

Siting implications generated from those rows (REPORT.md): low-wind fires
defeat corroboration rather than detection (one node smells it, no second
ever does); the 12-node ring has a hard radius of roughly a kilometre in
near-calm; the eastern ridge sector has no detection redundancy; in-town
starts are reported by humans first, so the mesh's value there is the
post-alarm products; at 14 m/s the cascade still landed 7.5 min before the
already-fast authored call.

## 6. Stress testing (Phase 13)

Six tests, all passing, encoding measured behaviour. Two designed limitations
were found and pinned rather than hidden:

| case | measured outcome |
|---|---|
| (a) two simultaneous ignitions | physics grows two independent fronts; the protocol corroborates both into **one incident**, and **DETECT packets arriving after the Tier-2 cascade are dropped at the source by storm suppression**, so a second fire reported post-cascade never reaches the head and Foresight is never pointed at it |
| (b) 7 of 12 nodes fail mid-scenario, including the head | a new head self-elects, the 5-node rump still corroborates to Tier 2, and the cascade reaches every survivor |
| (c) ignition inside the town grid | urban cells accept ignition and spread measurably slower than the timber ridge under identical wind; matches the town_origin_fire hindcast (−6.5 min) |
| (d) all nodes degraded | the distinct-origin gate (weight × confidence ≥ 0.3) means floor-weighted (0.2) reports max out at 0.198, so **an all-degraded mesh can never reach Tier 1 or 2**; Tier-0 chirps still sound locally |

Neither limitation was fixed in Round 2; REPORT.md explains why (incident
disambiguation is protocol surgery; health gating's siren-deafness is the
documented cost of refusing to cry wolf, and argues for fleet-health
maintenance alerts).

## 7. What the numbers do not say

- Nothing here measures a real sensor, a real radio, real smoke or a real
  fire. See [04_GAP_REGISTER.md](04_GAP_REGISTER.md).
- The surrogate can only be as right as its Rothermel-scope teacher.
- The confounder library is authored, so the detection F1 is a bound on the
  problem as the team defined it.
- The hindcast "warning minutes" are relative to a 911 minute the team typed
  into a YAML file.

---

Previous: [02_DATA_PIPELINE.md](02_DATA_PIPELINE.md). Next:
[04_GAP_REGISTER.md](04_GAP_REGISTER.md). Pitch figures rendered from these
numbers: [PITCH_ASSETS.md](../PITCH_ASSETS.md).
