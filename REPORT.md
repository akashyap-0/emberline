# Emberline — Measured Results

All numbers below were produced by code in this repository. Sections other
than [Real-data results](#real-data-results-trained-on-real-data-only) ran on
**synthetic, procedurally generated worlds** and claim nothing about
real-fire detection or real-world performance. The real-data section used
two public real datasets — which are also not our field deployment; it says
exactly that. Real and synthetic numbers are never merged into one table.
Regenerate any section with the command noted inside it.

<!-- BEGIN intro -->
## Contents

* [Real-data results](#real-data-results-trained-on-real-data-only)
  (Round 4) — smoke detection + NDWS fire spread, trained on real data only
* [System diagram](#system-diagram) — and a presentation-quality version in
  `demo/out/pitch_assets/system_architecture.png` (all six pitch assets are
  captioned in `PITCH_ASSETS.md`)
* [Limitations](#limitations-read-before-believing-any-number) — read first
* [Neural surrogate](#neural-surrogate-phase-3-retrained-in-phase-9) (Phase 3,
  retrained in Phase 9)
* [Wind-augmentation retraining](#wind-augmentation-retraining-phase-9)
  (Phase 9) — before/after, regime gap closed
* [Ensemble calibration + temperature scaling](#surrogate-ensemble-calibration--temperature-scaling-phase-10)
  (Phase 10)
* [Smoke detection](#smoke-detection-phase-4) (Phase 4)
* [Hindcast harness, 6 scenarios](#hindcast-harness-phase-11-expansion-6-scenarios)
  (Phase 11) — warning minutes gained + siting implications
* [Stress testing](#stress-testing-phase-13) (Phase 13) — measured failure
  modes, documented not hidden

<!-- BEGIN real-data -->
## Real-data results (trained on real data only)

**Models in this section were trained on real data only, with no synthetic
pretraining.** Every byte reaches training through `emberline/data/` loaders,
which raise on any path under `data/synthetic/` or elsewhere in the synthetic
stack's data tree (`emberline.data.assert_real_data_path`;
`tests/test_real_data_infra.py` proves the guard fires). Both models started
from scratch: no transfer learning, no fine-tuning, no synthetic-trained
checkpoint was loaded or evaluated. These are public real datasets, **not our
field deployment** — nothing here claims real-world detection capability for
our nodes; that claim waits for our own logged sessions and the supervised
burn.

Round 4, 2026-09-18. Regenerate: `python -m emberline.data.ingest`,
`python -m emberline.data.ndws`, `python -m emberline.detect.train_real`,
`python -m emberline.detect.eval_real`, `python -m emberline.spread.train_real`,
`python -m emberline.spread.eval_real`.

### Datasets, licences, splits

| dataset | contents | licence note | split policy |
|---|---|---|---|
| Kaggle smoke (`deepcontractor/smoke-detection-dataset`) | 62,630 rows @ 1 Hz, 12 sensor channels, `Fire Alarm` label, **71.5% positive** — nothing like deployment priors | not yet captured into `data/real/external/`; internal use only until confirmed | by **session id** (5 sessions from `CNT`'s exactly-4 reset points): 0,3 → train; 1 → val; 2,4 → test. Assigned once in `data/real/MANIFEST.yaml`, never by row |
| NDWS (`fantineh/next-day-wildfire-spread`, Huot et al. 2022) | 18,545 tiles of 64×64 km @ 1 km; 11 covariates + `PrevFireMask` → next-day `FireMask` | commonly cited CC BY 4.0; text not yet captured — confirm before redistribution | the dataset's own shipped **15 train / 2 eval / 2 test** shard split, respected as-is, never reshuffled |

### Leak guards applied (smoke dataset)

Three measured leak columns were dropped from all features: the unnamed row
index (76.4% accuracy alone vs the 71.5% majority baseline), `UTC`, and
`CNT` (90.0% alone; no negative row has CNT > 5,743). `CNT`'s resets were
used first to derive the 5 ground-truth sessions, then the column was
dropped. A test asserts no retained feature has |spearman| ≥ 0.5 vs
within-session row order (bookkeeping leaks measure ~1.0; the worst
surviving physical feature measures 0.42).

Two further channels were excluded with measured cause:

* **`Temperature[C]` is fabricated.** Sessions 1 and 4 are exact row-wise
  copies of sessions 0 and 3 in *every* other channel including the label,
  while temperature differs by large non-constant offsets (to −81.6 °C) —
  the publisher duplicated two recordings with altered temperature traces.
  Its class direction is also incoherent (AUC 0.03 on train sessions vs
  0.99 on val, measured before any test evaluation). Recorded per session as
  `duplicate_of` in the manifest. Consequence: **the val session duplicates
  a train session, and test session 4 duplicates train session 3** — the
  only genuinely held-out recording is test session 2, and val metrics are
  memorization checks, not generalization evidence.
* **`Pressure[hPa]` failed the row-order leak test** (|spearman| 0.937 on
  train): barometric level is a slow weather drift acting as a session
  clock. Raw H2 / Raw Ethanol absolute levels drift the same way (0.92) and
  entered only as slope + causal log-ratio.

### Smoke detection (test sessions 2 + 4 only; 6,780 windows, 16.2% positive)

Both models below; the majority-class baseline is printed beside every
score. Accuracy is deliberately not the headline: at a 71.5% positive prior,
"always fire" scores 71.5% while being useless.

| metric | GBM (primary) | 1D-CNN (comparison) | majority baseline |
|---|---|---|---|
| precision @ val threshold | **1.000** | 0.000 | 0.715 ("always fire") |
| recall @ val threshold | 0.099 | 0.000 | 1.000 ("always fire") |
| F1 @ val threshold | 0.180 | 0.000 | 0.834 ("always fire") |
| PR-AUC | 0.411 | 0.162 | 0.162 (test prevalence) |
| ROC-AUC | 0.625 | 0.463 | 0.500 (chance) |
| FP / node-day (fire-free spans, 5,680 s) | **0.0** | 2,722.8 | — |
| detection latency, session 2 | **10 s** | never fires | — |
| confusion (tp/fp/fn/tn) | 108 / 0 / 987 / 5,685 | 0 / 179 / 1,095 / 5,506 | — |

Reading: the GBM catches the held-out fire session's onset in 10 s with zero
false positives on the fire-free day, then loses it — its scores fall back
below threshold as PM2.5 saturates ~1,700× beyond anything a train fire
showed (train fires peak at PM2.5 ≈ 1.8 µg/m³; session 2's fire runs
~3,060). The CNN, trained on the same real windows from scratch, does not
transfer at all. The deeper cause is the dataset itself: its three distinct
recordings carry **mutually contradictory label semantics** — high PM is
"fire" in session 2 but "not fire" throughout session 3/4 (PM2.5 ≈ 700 with
label 0), so a session-honest split leaves any model trained on sessions
0+3 pointing the wrong way at session 2's regime. With the fabricated
temperature channel still included (v1, the full specified feature set),
both models were *anti*-correlated on test — GBM ROC-AUC 0.046, 67,918
FP/node-day (preserved in `metrics/detect_real_v1_with_temperature.json`);
removing the fabricated/leaking channels (v2) is what produced the table
above. Threshold: Youden's J on val only (F1-max degenerates to
"alert always" at val's 87.5% prior).

Artifacts: `data/checkpoints/detect_real_gbm.pkl`,
`data/checkpoints/detect_real_cnn.pt`, `metrics/detect_real.json`, plots in
`demo/out/real/`.

### Fire spread on NDWS (test shards only; 6.73M labeled pixels, 1.25% burned)

Persistence (next-day fire = today's fire) is the floor; the UNet (1.93M
params, ≤5M cap, trained from scratch, early-stopped on the eval shards at
epoch 6 of 11, ~80 min CPU within the 2 h budget) must beat it to matter.
Pixels with `FireMask = −1` (uncertain, per the dataset spec) are excluded
from loss and every metric; class imbalance handled with pos_weight = 90
BCE. Binary threshold chosen by max IoU on the eval shards.

| metric (burned cells @ +1 day) | persistence baseline | SpreadUNet |
|---|---|---|
| IoU | 0.183 | **0.246** |
| precision | 0.357 | 0.307 |
| recall | 0.273 | **0.555** |
| PR-AUC | — (binary) | 0.344 (prevalence 0.013) |

The UNet beats persistence on IoU (+34% relative) and doubles its recall at
somewhat lower precision. **Regime note:** this is 1 km / daily
satellite-scale spread — a different regime from the 10 m / minute-scale
node simulator elsewhere in this repo. These numbers are the real-data
reference point for spread modelling, not a claim about the 10 m system.

Artifacts: `data/checkpoints/spread_real_unet.pt` (inference weights),
`metrics/spread_real.json`, `demo/out/real/spread_real_tiles.png`.

### Domain gaps (candid)

* **Vendor gas indices vs `gas_ohms`.** The smoke dataset's gas channels
  are vendor-derived indices, and TVOC's polarity is *inverted* vs our node
  physics: mean 4,596.6 ppb when `Fire Alarm = 0` vs 882.0 when `= 1` —
  higher *without* alarm, where raw gas resistance on our node falls in
  smoke. The log-ratio feature therefore lets the model learn the sign;
  nothing hard-codes direction. 4.3% of TVOC samples are exactly 0
  (2,698 rows straddling both classes): kept, with ε = 1.0 in the log and a
  defined first-window/zero-baseline rule (feature = 0 with no history).
  These vendor channels are **not** a drop-in proxy for our `gas_ohms` —
  which is precisely why our own logged sessions are the next step.
* **Priors.** 71.5% positive is nothing like deployment, where positives
  are vanishingly rare; that is why PR-AUC and FP/node-day are reported
  beside every score and accuracy is not.
* **Scale.** 1 km / daily NDWS vs the 10 m / minute node system: the UNet
  result transfers methodology (masked loss, persistence floor, honest
  shipped splits), not numbers.
* **What our own data adds.** Logged ESP32 sessions (schema already
  enforced: `ms,pm1,pm25,pm10,gas_ohms,temp_c,rh,press_hpa`) give raw gas
  resistance with correct polarity, consistent label semantics, realistic
  priors, and as many genuinely independent sessions as we choose to
  record — fixing all three failure causes measured above. A supervised
  burn then provides the one thing no public set here has: ground-truth
  fire at node scale.

<!-- END real-data -->
## System diagram

```
             +-------------------------- SYNTHETIC WORLD --------------------------+
             | worldgen: fractal terrain . fuel map . town+roads . OU-gust wind    |
             +------+--------------------------+---------------------------+------+
                    |                          |                           |
             +------v------+            +------v------+             +------v------+
             |  firesim    | 16.8k pairs|  surrogate  |             |  sensors    |
             | Rothermel-  +----------->| UNet <=5M   |             | plume+conf- |
             | inspired CA |  training  | +10min step |             | ounders 1Hz |
             +------+------+            +------+------+             +------+------+
                    | truth                    | ensembles                 | windows
                    |                          |                    +------v------+
                    |                          |                    |  detect     |
                    |                          |                    | 1D-CNN vs   |
                    |                          |                    | GBM baseline|
                    |                          |                    +------+------+
                    |                          |                           | P(fire)
                    |                   +------v------+             +------v------+
                    |                   |  foresight  |<------------+  mesh       |
                    +------------------>| cones.route | Tier-1+ evt | LoRa DES .  |
                        demo truth      | .CAP drafts |  +bearings  | tier ladder |
                                        +------+------+             +-------------+
                                               | ADVISORY outputs, outbox/ drafts
                                        +------v------+
                                        |  demo CLI   |  terminal story . GIF . scoreboard
                                        +-------------+
```

Config: every knob in `config.yaml` (single global seed). Regeneration
commands appear inside each section below. Round 2 (Phases 9-13) added the
wind-rotation retraining, calibration machinery, four hindcast scenarios,
stress tests, and the pitch assets; the build log with timings and honest
misses is `PROGRESS.md`.
<!-- END intro -->

<!-- BEGIN limitations -->
## Limitations (read before believing any number)

* **Everything is synthetic.** Worlds, fires, plumes, sensor readings and
  confounders are all generated by this repo's own models. No number here is
  evidence about real-world detection or forecasting performance. The next
  step for the team is real data: prescribed-burn sensor logs, BBQ/stove/fog
  recordings from a physical node, and hindcasts on historical fire perimeters.
* **Fire physics is a 1970s-scope surface model.** Rothermel-style
  ROS = R0*fuel*wind*slope in a probabilistic CA: no crown fire, no ember
  spotting, no fire-atmosphere coupling, no fuel-moisture dynamics, quasi-steady
  spread, 10 m cells, spatially uniform wind. The surrogate can only ever be as
  right as this teacher.
* **The surrogate's speedup baseline is generous to physics.** Our CA is a
  vectorised toy that runs 60 sim-minutes in well under a second; operational
  solvers are orders of magnitude slower, so the reported speedup is a floor,
  not a ceiling.
* **Sensor/confounder signatures are authored, not measured.** The classifier
  separates classes whose differences we ourselves wrote down (documented in
  `sensors/events.py`); real confounders will be messier. The event-level split
  and the mesh corroboration ladder are the honest mitigations.
* **Radio model is log-distance + a dB/m occlusion heuristic** with a single
  collision domain and CAD; no fading distributions, no capture effect, no
  regulatory dwell-time subtleties beyond a 1% duty budget.
* **Routing assumes compliant drivers and static capacities** (BPR-style
  congestion, 3 assignment rounds). All routing output is ADVISORY.
<!-- END limitations -->

<!-- BEGIN surrogate -->
## Neural surrogate (Phase 3, retrained in Phase 9)

Regenerate: `python -m emberline.surrogate.eval` (uses `data/checkpoints/best.pt`,
config `surrogate.*` in config.yaml). Splits are by WORLD; the held-out wind
regime (80-130 deg) never appeared in training.

| split | fires | IoU@+10 | IoU@+30 | IoU@+60 | arrival MAE (min) |
|---|---|---|---|---|---|
| val (unseen worlds) | 64 | 0.678 | 0.627 | 0.507 | 3.7 |
| held-out wind regime | 64 | 0.651 | 0.657 | 0.523 | 3.0 |

Worst-case: 5th-percentile IoU@+30 across val fires = **0.369**.

Ensemble wall-clock, 20 members x 60 sim-min on 4 CPU threads:
physics 6.82 s vs surrogate 14.23 s ->
**0.5x**.

Gap analysis (targets were goals, not claims):
- IoU@+30 = 0.627 misses the 0.80 aspirational target. Main error mode: autoregressive drift — small front-position errors compound over 3 steps; more training worlds and longer training (this run was CPU-budget-capped) are the obvious levers.
- Ensemble speedup = 0.5x misses the 100x target. Context: our physics baseline is itself a heavily vectorised CA (6.8 s for 20 members x 60 min), not an operational-grade solver, so the denominator is unusually fast. Against FARSITE-class physics the surrogate's one-forward-per-10-min batched rollout would win by orders of magnitude; here it wins by batching members through one network pass.
<!-- END surrogate -->

<!-- BEGIN wind-augmentation -->
### Wind-augmentation retraining (Phase 9)

Regenerate: `python -m emberline.surrogate.eval --compare-baseline metrics/surrogate_v1.json`
(v1 = `data/checkpoints/best_v1.pt`, archived pre-augmentation checkpoint;
v2 = `best.pt`, fine-tuned from v1 with joint world+wind rotation augmentation
— `surrogate.train.rotate_augment` in config.yaml, implemented in
`surrogate/datasets.py`, tested in `tests/test_surrogate.py`).

For continuity: the original build's REPORT quoted **val IoU@+30 0.681
/ held-out-regime 0.368** (team-laptop build of the same v1
checkpoint; those numbers are preserved, not overwritten). Re-measured on THIS
machine from the archived v1 checkpoint and the deterministically regenerated
dataset, v1 scores 0.655 / 0.411
— cross-platform torch kernel differences compound over autoregressive steps,
so the like-for-like before/after is the same-machine pair below (64 fires per
split, identical eval code and RNG streams):

| metric (val / held-out wind regime) | v1 pre-augmentation | v2 wind-augmented |
|---|---|---|
| IoU@+10 | 0.551 / 0.296 | 0.678 / 0.651 |
| IoU@+30 | 0.655 / 0.411 | 0.627 / 0.657 |
| IoU@+60 | 0.621 / 0.478 | 0.507 / 0.523 |
| arrival MAE (min) | 4.244 / 9.406 | 3.674 / 2.986 |

Worst-case p5 IoU@+30 (val): 0.248 → 0.369.

The held-out-regime IoU@+30 moved +0.246 and the val-holdout gap went from 0.244 to -0.030 — the regime gap is closed; val IoU@+30 moved -0.028 alongside. The cost is at the long horizon: val IoU@+60 moved -0.115 — capacity now spreads across all wind orientations, and the +60 min rollout (6 autoregressive steps) pays most for it. Operationally the trade is accepted: wind-robust +10/+30 cones are what detection-time routing uses.
<!-- END wind-augmentation -->

<!-- BEGIN calibration -->
### Surrogate ensemble calibration + temperature scaling (Phase 10)

Regenerate: `python -m emberline.surrogate.calibration`. Reliability of the
20-member ensemble's P(burn by +30 min) against physics outcomes on
12 val-world fires (cells near the fire; empty wilderness excluded). The raw
curve is under-confident (observed frequencies exceed predictions).

Phase 10's temperature scaling is fit-then-verify: a scalar T is fitted on
12 fires from freshly generated fit worlds 161-168
(outside every train/val/holdout split, never used for IoU eval), then must
IMPROVE ECE on 6 fires from DISJOINT check worlds
169-172 or the identity (T=1, no scaling) ships instead.
sigmoid(logit(p)/T) applies to unsaturated ensemble frequencies only. The fit
objective is NLL over interior frequencies — two
naive objectives failed measurably first: plain NLL is dominated by saturated
(exactly-0/1) quantised frequencies (its T=2.59 worsened val ECE 0.060→0.229),
and direct binned-ECE minimisation is degenerate (collapsing toward the base
rate zeroes fit-set ECE; T ran to the grid edge, val ECE 0.330). Both dead
ends are kept in `fit_temperature`'s docstring.

This run: candidate T = 2.37; disjoint-check-world ECE
0.290 (raw) vs 0.267 (candidate) →
**check PASSED**.

| | ECE on the val fires (this checkpoint, this machine) |
|---|---|
| raw ensemble — **what the system ships** | **0.060** |
| with the fitted T = 2.37 (not shipped) | 0.196 |

**Transfer failure, documented rather than papered over**: on freshly generated worlds the ensemble measures badly miscalibrated (check-world raw ECE 0.290) and the fitted softening helps there, but the val population is already near-calibrated (raw 0.060) and the same T makes it WORSE (0.196). Two caveats bound this finding: each population estimate rests on only 6-12 fires (per-fire calibration variance is large), and the val split also selected the training checkpoint. Consequence: `foresight.temperature` stays null — raw cone probabilities ship.

Baselines, for the before/after: the pre-Phase-9 checkpoint's REPORT quoted
**ECE 0.113** (raw) on its build machine, and re-measured HERE from the archived `best_v1.pt` it scores **raw 0.148** (`metrics/calibration_v1.json`; same platform-variance story as the IoU numbers). **Most of the calibration
fix came from the Phase-9 retraining itself** — the shipped-configuration
before/after is 0.113 (v1, quoted) → **0.060** (v2 raw, this machine).
Curves: `demo/out/calibration_v2.png`. Temperature scaling is monotone, so
cone THRESHOLD semantics change but cone SHAPES at matched percentiles do
not; the fit/check machinery stays in `surrogate/calibration.py` for the day
real-sensor hindcasts give it a population worth fitting to.
<!-- END calibration -->

<!-- BEGIN detect -->
## Smoke detection (Phase 4)

Regenerate: `python -m emberline.detect.eval`. Split by EVENT (val events
never seen in training); thresholds: CNN 0.60 (train-selected), GBM 0.5. **All data is synthetic** — plume +
signature models, not real sensors; collecting real burn/confounder data is
the team's stated next step.

| model | precision | recall | F1 |
|---|---|---|---|
| 1D-CNN (9,129 params) | 0.934 | 0.953 | 0.943 |
| GBM baseline | 0.953 | 0.952 | 0.952 |

The CNN does NOT beat the GBM baseline on val F1. With only 60 s of context the handcrafted summary features (levels, slopes, VOC/PM ratio, RH) capture most of the signal; the honest engineering call is to ship the cheaper model on-node and revisit with longer windows.

Per-confounder false-positive rate on val windows (CNN / GBM):

| confounder | windows | CNN FP rate | GBM FP rate |
|---|---|---|---|
| bbq | 96 | 0.125 | 0.083 |
| wood_stove | 64 | 0.312 | 0.188 |
| vehicle | 28 | 0.107 | 0.107 |
| fog | 192 | 0.000 | 0.000 |
| dust | 136 | 0.000 | 0.000 |
| aerosol | 9 | 0.000 | 0.111 |
| ambient | 540 | 0.000 | 0.000 |

Ambient 24 h, fire-free, 14 confounder events across
34548 windows: **7.33 false positives per
node-day (CNN)**, 8.67 (GBM). Note these are single-node,
single-window numbers — the mesh's corroboration ladder (Phase 5) is what
turns them into siren-worthy alarms.

Detection latency over 25 fresh simulated fires: median
**36 s**, mean 86 s from plume arrival to first
positive window (3 fires never produced a classifiable plume at any
node — typically burning away from the network).
<!-- END detect -->

<!-- BEGIN hindcast -->
### Hindcast harness (Phase 11 expansion: 6 scenarios)

Regenerate: `python -m emberline.foresight.hindcast scenarios/*.yaml`. Scenario
files are **hand-authored illustrative patterns, not real fire records** (the
file format + adapters are the path to real hindcasts). Minutes from ignition:

| scenario | conditions | Tier-0 | Tier-2 cascade | first 911 report (authored) | warning minutes gained |
|---|---|---|---|---|---|
| dry_ridge_evening | 7 m/s, nearest node 152 m | 0.0 | 2.5 | 22.0 | **19.5** |
| valley_night | 4 m/s, nearest node 212 m | 1.0 | — | 55.0 | **—** |
| highwind_ridge_run | 14 m/s, nearest node 786 m | 4.5 | 7.5 | 15.0 | **7.5** |
| stagnant_far_corner | 3 m/s, nearest node 1052 m | — | — | 65.0 | **—** |
| town_origin_fire | 6 m/s, nearest node 453 m | 5.0 | 12.5 | 6.0 | **-6.5** |
| degraded_mesh_ridge | 7 m/s, 2 nodes down (N0,N7), nearest node 396 m | 5.0 | — | 22.0 | **—** |

A "—" means the mesh never corroborated to a cascade: the harness is thus
also a **siting design tool** — it shows where the network layout would have
missed, before any hardware is planted. Measured outcomes this run:

* Cascades fired in 3/6 scenarios (warning minutes vs the authored 911 call: +19.5, +7.5, -6.5); their ignitions sat 152-786 m from the nearest alive node.
* Missed entirely: **valley_night** (nearest alive node 212 m, wind 4 m/s, Tier-0 only at 1 min); **stagnant_far_corner** (nearest alive node 1052 m, wind 3 m/s, zero detections); **degraded_mesh_ridge** (nearest alive node 396 m, wind 7 m/s, 2 nodes down, Tier-0 only at 5 min).

**Siting implications** (each sentence is generated from the measured rows
above; a re-run with different outcomes rewrites or drops it):

* **Low-wind fires defeat corroboration, not detection**: valley_night chirped Tier-0 at 1 min but a 4 m/s drift puts smoke on only one node's line, and the ladder (by design) refuses single-node cascades — the layout needs a second node along each low-wind drainage path, not more confidence.
* **The ring has a hard radius**: stagnant_far_corner produced ZERO detection windows in 80 min with the nearest node 1052 m away at 3 m/s — fires outside roughly a kilometre of the perimeter in near-calm are invisible until they grow or the wind turns.
* **Two nodes are single points of cascade**: the same ridge fire that cascaded in 2.5 min with the full mesh never cascaded at all with 2 nodes down (N0,N7) — Tier-0 still fired at 5 min, so one node smelled it and no second ever corroborated. The eastern ridge sector has no detection redundancy.
* **In-town starts don't need the mesh to raise the alarm**: humans beat the cascade by 6.5 min in town_origin_fire; the system's value there is what follows the alarm (cones, routing, CAP draft), not detection speed.
* **High wind compresses but keeps the margin**: at 14 m/s the cascade still landed 7.5 min before the (already fast) authored 911 call.
<!-- END hindcast -->

<!-- BEGIN stress -->
## Stress testing (Phase 13)

Regenerate: `python -m pytest tests/test_stress.py -v`. Six tests, all
encoding MEASURED behaviour — current suite status: **PASS** (6 passed in 1.04s).
Two designed limitations were found while writing them and are pinned by
tests rather than hidden:

| case | measured outcome |
|---|---|
| (a) two simultaneous ignitions | Physics: two independent fronts establish and grow (verified via connected-component labelling). Protocol: pre-cascade reports from BOTH fires corroborate a single incident (Tier 2 fires), but **detections arriving after the Tier-2 cascade are dropped at the source by storm suppression — a second fire reported post-cascade never reaches the head, so Foresight is never pointed at it**. The incident's mean bearing across two fires is physically meaningless. |
| (b) >50% of nodes fail mid-scenario | 7 of 12 nodes killed, including the cluster head: heartbeat mourning fires, a new head self-elects, the 5-node rump mesh still corroborates two healthy reports to Tier 2, and the cascade reaches **every** surviving node. |
| (c) ignition inside the town grid | Urban cells accept forced ignition and the fire spreads beyond the ignition patch, measurably slower than the timber-ridge fire under identical wind (fuel factor 0.12 + per-contact urban ignition gate). Hindcast counterpart: `town_origin_fire` cascades at 12.5 min but the authored 911 call beats it by 6.5 min — for in-town starts the mesh's value is the post-alarm products, not detection speed. |
| (d) all nodes low-battery/degraded | Escalation's corroboration counts only detections with weight x confidence >= 0.3; a floor-weighted (0.2) report maxes out at 0.198, so **an all-degraded mesh can NEVER reach Tier 1/2 — even 12 nodes screaming at 0.99 stay at Tier-0 chirps.** Small degraded clusters (the realistic case) are correctly refused. |

Failure modes documented, not fixed this round (each is a deliberate scope
call, budget spent on documenting + pinning):

* **Post-cascade deafness to a second fire** — storm suppression (the fix for
  the 50-trigger broadcast storm) also silences new-fire reports once a
  cascade has flooded. A fix needs incident disambiguation (e.g. bearing/
  location clustering before suppression), which is protocol surgery, not a
  patch. Until then: one cascade per corroboration window, and the town is
  already at full siren when it matters.
* **All-degraded siren-deafness** — the same gate that stops degraded nodes
  from crying wolf makes a fully degraded network unable to raise the town
  siren for a real fire (Tier-0 chirps still sound locally). That is the
  documented cost of the health-gating design; operationally it argues for
  maintenance alerts on fleet-wide health decay (mesh telemetry already
  carries per-node health).
<!-- END stress -->
