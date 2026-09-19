# Emberline build log

## 2026-09-18 — ROUND 4: REAL DATA ONLY (detection + spread on real datasets) ✅

Environment: same team Windows 11 laptop, CPU only, `bash verify.sh` (no
make). Branch `main`. The one rule of the round: **every model trained
tonight uses exclusively real data** — no synthetic samples in any split, no
synthetic-pretrained checkpoint, no evaluation of synthetic-trained models.
Enforced in code: all new training/eval loads through `emberline/data/`,
whose loaders raise on any path under `data/synthetic/` (or elsewhere in the
synthetic stack's data tree); tests prove the guard fires. The synthetic
stack is untouched and its tests stay green at every commit.

Phase 0 — cleanup + safety: deleted the two junk files from a terminal
redirect accident (never tracked, one was a `less` help page); committed the
verify-refresh of `metrics/demo_last_run.json`; `bash verify.sh` green;
pushed the prep commits. `data/real/INVENTORY.md` (supersedes the prelim):
all 20 raw files with size, format, SHA-256 — identity by digest, never size
(16 of 19 NDWS shards are byte-identical in size). Licences honestly marked
"not yet captured into external/" rather than asserted from memory.

Phase 1 — `emberline/data/`: `schemas.py` (Kaggle CSV as it truly is,
unnamed index column included; NDWS tf.train.Example feature spec; the
future ESP32 logger contract `ms,pm1,pm25,pm10,gas_ohms,temp_c,rh,press_hpa`
— schema only, validators reject unknown/missing columns, never coerce);
`tfrecord_lite.py` — a pure-Python TFRecord + Example parser that reads all
19 shards (18,545 records), so **tensorflow was NOT added** (constraint 6
resolved the light way, with round-trip tests); `ingest.py` CLI — validates,
writes one parquet per smoke session to `interim/`, and writes
`MANIFEST.yaml` (path, sha256, bytes, source, licence note, split of every
training group; idempotent, hash cache keyed by size+mtime). Split policy in
the manifest, assigned once: smoke by session id (CNT's exactly-4 resets →
5 sessions: 0,3 train / 1 val / 2,4 test), NDWS by its shipped 15/2/2 shard
split. 15 tests (guard, schema rejection, determinism, split stability).
Dependencies added and recorded: pandas, pyarrow.

Phase 2 — smoke detection on real data, and two data-quality discoveries
that shaped everything:
(a) **the Temperature channel is fabricated** — sessions 1 and 4 are exact
row-wise copies of sessions 0 and 3 in every other channel *including the
label*, with temperature altered by non-constant offsets to −81.6 °C; so
val duplicates train, test session 4 duplicates train session 3, and only
test session 2 is genuinely held out (recorded as `duplicate_of` in the
manifest);
(b) **ambient levels are session clocks** — pressure mean hit |spearman|
0.937 vs row order on train, H2/ethanol levels 0.92; the mandated leak test
(ceiling 0.5, every feature) forced them out, leaving dynamics (slopes,
causal log-ratios) and particulate levels. The three known leak columns
(row index / UTC / CNT) were dropped after deriving sessions from CNT.
Models from scratch: GBM (primary; hyperparameters chosen on val — column
subsampling is what stops session-ambient memorization) and the SmokeCNN
architecture re-instantiated at the real channel count. Thresholds via
Youden's J on val only (F1-max degenerates at val's 87.5% prior). Test
(sessions 2+4): GBM precision 1.000 / recall 0.099 / PR-AUC 0.411 /
ROC-AUC 0.625, **0.0 FP/node-day** on 5,680 fire-free s, **10 s detection
latency** on the held-out fire; CNN does not transfer (0 recall,
2,723 FP/day). v1 with the fabricated temperature channel included was
*anti*-predictive on test (GBM ROC-AUC 0.046) — preserved in
`metrics/detect_real_v1_with_temperature.json` beside the v2 numbers; the
dataset's three distinct recordings carry mutually contradictory label
semantics (high PM = fire in session 2, = not-fire in sessions 3/4), which
is the measured argument for our own logged sessions. 6 more tests.

Phase 3 — NDWS spread: shards → per-shard resumable float16 `.npy` tiles in
`interim/` (wind-direction garbage values, observed to −465,922°, clipped to
[0, 360] at conversion — documented); `SpreadUNet` 1,931,009 params (cap
5M, asserted in a test), masked weighted BCE (FireMask −1 excluded,
pos_weight 90), early stop on the eval shards (epoch 6 of 11, ~80 min CPU,
inside the 2 h cap, checkpoint-resume exercised for real after the
background task was killed). Test shards, 6.73M labeled pixels:
**UNet IoU 0.246 vs persistence 0.183** (+34% relative), recall 0.555 vs
0.273, PR-AUC 0.344 at 1.25% prevalence. Explicitly a 1 km/daily
satellite-regime reference point, not a claim about the 10 m system.
6 more tests.

Phase 4 — REPORT.md gained the top-level "Real-data results (trained on
real data only)" section (detection + spread tables, each with dataset,
licence note, split policy, leak guards, majority/persistence baselines
beside every model score, and the candid domain-gaps paragraph; real and
synthetic numbers never share a table); `data/real/README.md` now carries
the 4-step add-a-new-dataset recipe; suite 88 tests, `bash verify.sh` green
including all synthetic tests.

## 2026-09-16 — ROUND 3: MVP PACKAGE (documentation + one-command entrypoint) ✅

Environment: team Windows 11 laptop (12 cores, Python 3.13, torch 2.6 CPU),
Git Bash; `make` is not installed here, so `bash verify.sh` was used and the
new entrypoint falls back to it automatically. Branch `emberline-build`
recreated from `main` (`dc94489`, the consolidated Round-2 state). Started by
running `verify.sh` on the untouched checkout: 54 tests green, demo smoke
48 s, scoreboard identical to REPORT.md.

Rules kept this round: **no model, training or simulation code changed**; no
number invented (every figure in `docs/` was copied from REPORT.md, the
metrics JSON, or a run performed here, and a script cross-checked every
numeric token in `docs/` against those sources); everything physical or
commercial is labelled PLANNED / NOT BUILT; one commit per phase, repo
runnable at each.

Phase 1 — `docs/AUDIT.md`: module-by-module inventory (path, purpose, key
symbols, tests, REPORT metrics produced, state: measured /
implemented-tested / implemented-untested / stub) plus the committed-artifact
and test inventories. Findings worth recording: `adapters/` are three stubs;
`foresight/economics.py` has no section in the current REPORT.md and its
metrics file is not committed; the 12-node demo mesh is fully connected at
the configured path-loss exponent (cascade max 1 hop), so multi-hop relaying
is exercised only by tests with a dense-canopy exponent.

Phase 2 — the document set in `docs/`: `00_OVERVIEW` (Lahaina thesis,
escalation ladder, headline numbers, Mermaid system diagram),
`01_ARCHITECTURE` (every module in execution order with equations quoted
from docstrings and config keys; sequence diagram of one ignition from first
sample to CAP draft), `02_DATA_PIPELINE` (surrogate pairs, detection windows,
scenario files, and the PLANNED real-data contract: 1 Hz CSV
`ms,pm1,pm25,pm10,gas_ohms,temp_c,rh,press_hpa`, real features, event-labelled
sessions, separate real report), `03_RESULTS` (faithful REPORT.md summary
with misses stated: IoU@+30 0.627 < 0.80, speedup 0.5× < 100×, CNN 0.943 <
GBM 0.952), `04_GAP_REGISTER` (20 rows, Built-simulated / Partially built /
NOT BUILT, and a 10-step minimum path to a field pilot), `05_ROADMAP`
(now / next / later, each step with its proof artifact), `06_GLOSSARY`.
README rewritten as the front door (it had been UTF-16 on disk; now UTF-8).

Phase 3 — `make mvp` / `python -m emberline.mvp` (`emberline/mvp.py`): runs
`make verify` (or `bash verify.sh`), the canonical demo
(`--scenario ridgeline --fast --wind-shift 40 --kill-node N3`), the pitch
assets, and writes `docs/MVP_RUN.md` from the artifacts the run produced;
refuses to write if `metrics/demo_last_run.json` is missing, stale (older
than the demo start), or not from the canonical run. 7 tests in
`tests/test_mvp.py` (suite now 61). Measured here: whole command 125 s
(verify 72 s, demo 47 s, pitch assets 5 s), well inside the 10-minute budget;
61 passed; REPORT cross-checks all PASS.

Phase 4 — `docs/diagrams/`: the six Mermaid blocks extracted verbatim to
`.mmd` files (system, architecture_modules, escalation_ladder,
ignition_sequence, data_flow, roadmap) and `render_diagrams.py` (matplotlib
only; the repo has no Mermaid renderer) producing `system.png` and
`data_flow.png` for slides.

Phase 5 — consistency pass: numeric cross-check script over `docs/` (all
tokens traced to REPORT/PROGRESS/config/metrics/code or labelled team
context), link check (all relative links resolve), test-count references
updated to 61, final `make mvp` green, this entry, push.

Not done / out of scope (by instruction): no code behaviour changed, no
hardware or real data, no web UI. Everything in `docs/04_GAP_REGISTER.md`
remains open.

## 2026-09-01 ~06:45 UTC — ROUND 2 COMPLETE (Phases 9-14) ✅

Environment: fresh cloud container (4 cores, Linux). Started by regenerating
the gitignored training data from seed (byte-deterministic, 16,841 pairs) and
confirming `make verify` green on the committed state before touching code.

Headline measured results (details + caveats in each phase entry below and in
REPORT.md — quoted-vs-here differences are cross-platform torch variance,
both records preserved):
- **Phase 9**: joint world+wind rotation augmentation; held-out wind-regime
  IoU@+30 **0.411 → 0.657** (val 0.655 → 0.627): regime gap CLOSED
  (0.244 → −0.030). Honest cost: val IoU@+60 0.621 → 0.507. v1 checkpoint +
  metrics archived; nothing overwritten.
- **Phase 10**: temperature scaling (fit-then-verify on fresh disjoint
  worlds). Shipped-config ECE **0.113 (quoted) / 0.148 (v1 here) → 0.060 (v2
  raw)** — the retraining itself did the calibrating; the fitted T=2.37 fails
  val-population transfer and does NOT ship (documented).
- **Phase 11**: 6 hindcast scenarios; gains +19.5/+7.5 min, an honest −6.5
  (in-town start), and 3 misses that map the layout's blind spots (siting
  implications generated from measured rows).
- **Phase 12**: six 1920×1080 pitch PNGs, every number from real runs,
  committed; PITCH_ASSETS.md has captions + regen commands.
- **Phase 13**: 6 stress tests passing; two real limitations documented and
  pinned (post-cascade deafness to a second fire; all-degraded mesh can never
  cascade).
- **Phase 14**: REPORT.md TOC + section reorder; final `make verify` green
  (54 tests + end-to-end smoke).

Round-2 rules kept: no Phase 1-8 regressions (verify green at every commit),
no fabricated numbers (every REPORT figure regenerates from a command), old
numbers preserved beside new ones, adapters/ untouched, no UI.

## 2026-09-01 ~06:30 UTC — ROUND 2, Phase 13: edge-case hardening ✅

6 new stress tests (tests/test_stress.py), all passing; REPORT.md gains a
"Stress testing" section that reports the measured outcomes INCLUDING the two
real limitations discovered while writing them (documented + pinned by tests,
deliberately not patched this round):
- (a) Two simultaneous ignitions: physics runs two independent fronts fine;
  the protocol corroborates BOTH fires into one incident and — the sharper
  finding — **post-cascade storm suppression drops later DETECT packets at
  the source**, so a second fire reported after the first cascade never
  reaches the head (Foresight is never pointed at it).
- (b) 7/12 nodes killed mid-scenario incl. the cluster head: self-heal
  re-elects, the 5-node rump still cascades, and delivery reaches every
  survivor.
- (c) In-town urban ignition: spreads (slower than wildland, as designed);
  ties to the town_origin_fire hindcast (−6.5 min vs 911).
- (d) All-degraded mesh: measured to be STRICTER than the docs implied — the
  distinct-origin gate (weight×conf ≥ 0.3) means floor-weighted (0.2) nodes
  can NEVER corroborate to Tier 1/2 at any count; chirps only. Flip side
  (siren-deaf degraded fleet) documented as the cost of health gating.

## 2026-09-01 ~06:15 UTC — ROUND 2, Phase 12: pitch-ready static assets ✅

Built `emberline/demo/pitch_assets.py` (`python -m emberline.demo.pitch_assets
[--only NAME]`) rendering six 1920×1080 PNGs to `demo/out/pitch_assets/`
(committed — whitelisted in .gitignore). Every number is LOADED from real-run
artifacts; missing sources refuse to render and print the regen command.
Data plumbing: the demo now dumps `demo/out/last_run_artifacts.{npz,json}`
(cones before/after shift, routing masks, plans, fire-state snapshots at each
forecast, node states, mesh log, wind vectors) and records kill/self-heal/
wind-shift timestamps in `metrics/demo_last_run.json`; verify.sh's smoke uses
the canonical `--kill-node N3 --wind-shift 40` so any verify regenerates
consistent artifacts.

Assets (all from the instrumented ridgeline run: Tier-0 30 s, Tier-1 120 s,
Tier-2 300 s, N3 killed t+390 s, self-heal t+570 s, wind shift t+870 s):
1. system_architecture — pipeline with live metrics (params from best.pt,
   F1s, IoU@+30 0.63/0.66).
2. cone_evacuation_before_after — +60 cone, live front, per-exit routes and
   loads across the +40° shift (6→0 road edges cut, stranded access point
   recovered).
3. mesh_topology — RSSI links, health-sized nodes, N3 kill + measured
   self-heal note, cluster head N0.
4. detection_confusion — per-confounder FP bars, CNN vs GBM.
5. calibration_before_after — v1 raw (0.148) vs shipped v2 raw (0.060),
   fitted-T transfer failure noted.
6. warning_timeline — the measured event timeline.
PITCH_ASSETS.md: one-line caption + exact regen command per asset.

## 2026-09-01 ~05:50 UTC — ROUND 2, Phase 10: calibration + temperature scaling ✅

Built: temperature scaling in `surrogate/calibration.py` — fit-then-verify on
FRESH worlds (fit 161-168, DISJOINT check 169-172; both outside every
train/val/holdout split), NLL over interior (unsaturated) ensemble
frequencies, saturated 0/1 frequencies pass through unscaled; optional
`foresight.temperature` hook (default null). 4 unit tests.

Two naive fit objectives failed measurably and are documented in code+REPORT:
plain NLL (saturation-dominated; T=2.59 worsened val ECE 0.060→0.229) and
direct binned-ECE (degenerate base-rate collapse; T→grid edge, val 0.330).

Measured (12 val fires, same rng stream as the original ECE):
- Headline shipped-config before/after: **ECE 0.113 (v1, quoted) → 0.060 (v2
  raw, this machine)** — the Phase-9 retraining itself did most of the
  calibrating. v1 re-measured here: raw 0.148.
- Honest finding: the cal-world-fitted T=2.37 passes its disjoint-world check
  (0.290→0.267) but does NOT transfer to the val population (0.060→0.196), so
  no scaling ships (`foresight.temperature: null`); population disagreement +
  small-sample caveats documented in REPORT.md. v1's candidate (T=0.83) was
  rejected by its check worlds.
- `demo/out/calibration_v2.png` (committed) carries both curves.

## 2026-09-01 ~05:05 UTC — ROUND 2, Phase 11: hindcast scenario expansion ✅

4 new hand-authored scenarios (all world 0, deterministic replays):
- `highwind_ridge_run` — 14 m/s SE approach, timber, 790 m from nearest node.
- `stagnant_far_corner` — 3 m/s night smoulder >1 km NE of the ring.
- `town_origin_fire` — urban-cell ignition INSIDE the town district.
- `degraded_mesh_ridge` — dry_ridge_evening's fire with N0+N7 dead at start
  (new harness support: `nodes_down:` kills nodes before head election).

Measured (all 6 in REPORT.md, warning-minutes-gained vs authored 911):
dry_ridge_evening **+19.5** (reproduces the original stretch number exactly),
highwind_ridge_run **+7.5**, town_origin_fire **−6.5** (humans beat the mesh
for in-town starts — honest negative), valley_night **—** (Tier-0 at 1 min,
never corroborates), stagnant_far_corner **—** (ZERO detections in 80 min),
degraded_mesh_ridge **—** (Tier-0 at 5 min, cascade never fires with the two
best-placed nodes dead). Harness now emits a **siting implications** block
generated from the measured rows (single-line-of-smell corroboration failure,
~1 km hard radius, no eastern-ridge redundancy, in-town value is post-alarm
products); metrics in `metrics/hindcast.json`.

## 2026-09-01 ~04:20 UTC — ROUND 2, Phase 9: wind-augmentation retraining ✅

Environment: fresh cloud container (4 cores). Regenerated the full training
set from seed — byte-deterministic: exactly 16,841 pairs again. Re-measured
the archived v1 checkpoint HERE before touching anything: val IoU@+30 0.655 /
holdout-regime 0.411 (vs 0.681 / 0.368 quoted from the laptop build — torch
CPU kernel differences compound over autoregressive steps; both records kept
in REPORT.md, nothing overwritten).

Built:
- `surrogate/datasets.py`: joint world+wind rotation augmentation — every
  training crop sampled through a rotated coordinate grid (order-0 for masks +
  fuel one-hot so they stay binary/one-hot, bilinear for elevation), wind
  vector rotated by the SAME matrix; `rotate_keep_frac` leaves 25% of samples
  on the native grid. Config: `surrogate.train.rotate_augment`.
- `surrogate/train.py`: `--init-from` (warm-start weights, fresh optimizer —
  the fine-tune path) and `--lr` override; resume semantics unchanged.
- 2 new tests: exact-90° world/wind consistency (rotated spread stays aligned
  with rotated wind, speed preserved) + rotated-item validity (one-hot, wind
  planes constant, monotone targets).
- `surrogate/eval.py --compare-baseline`: regenerates the Phase-9
  before/after REPORT section from metrics JSONs.

Run: fine-tune from best_v1.pt, lr 1e-3, early-stopped at step 2800 (best
crop val IoU@+10 0.9448 at step 1600, v1 was 0.9405), ~35 min wall on 4 cores.

Measured (64 fires/split, same eval code + RNG as v1 baseline):
- held-out wind regime IoU@+30: 0.411 → **0.657**; val 0.655 → 0.627;
  val-holdout gap 0.244 → **−0.030 — the regime gap is CLOSED.**
- val IoU@+10 0.551 → 0.678; worst-case p5 IoU@+30 0.248 → 0.369;
  holdout arrival MAE 9.4 → 3.0 min.
- Honest cost: val IoU@+60 0.621 → 0.507 (capacity spread across
  orientations; +60 rollout pays most). Documented in REPORT.md.

v1 checkpoint archived as `data/checkpoints/best_v1.pt` (committed) with its
metrics in `metrics/surrogate_v1.json`; old numbers remain reproducible.
`make verify` green (53 tests; demo smoke now uses the canonical
`--wind-shift 40` acceptance flags).

## 2026-08-30 05:40 UTC — Phase 1: World generation ✅

Built:
- `emberline/config.py` — YAML config loader + deterministic per-subsystem RNG streams
  (`rng_for(cfg, stream, world_id)`), single global seed.
- `emberline/worldgen/terrain.py` — spectral-synthesis fractal elevation (power ∝ k^-beta),
  slope components via central differences.
- `emberline/worldgen/fuel.py` — quantile-thresholded, elevation-biased fractal fuel map
  (water/grass/brush/timber/urban) with configurable area fractions.
- `emberline/worldgen/town.py` — flattest-site selection, street lattice (networkx),
  2–4 exit roads to the map edge, 150–400 building footprints with road access nodes.
- `emberline/worldgen/wind.py` — OU-gust wind model with piecewise-constant mean
  (regime shifts + runtime `apply_shift` for the demo's `--wind-shift`).
- `emberline/worldgen/__init__.py` — `World` dataclass + `generate_world(cfg, world_id)`
  (world_id>0 draws randomized wind regimes for surrogate training diversity).
- `emberline/viz.py` — hillshade/fuel/town PNG helpers.
- CLI: `python -m emberline.worldgen --preview` → `demo/out/worldgen/{terrain,fuel,town}.png`.

Fixed: repo's original README.md was UTF-16 which broke setuptools metadata; replaced with UTF-8.

Tests: 7/7 passing (`tests/test_worldgen.py`) — determinism, fuel fractions (±10%),
building count in range, road graph connected, every building reachable from an exit,
wind gust variance + shift semantics, distinct randomized worlds.

Canonical world 0: 315 buildings, 3 exits, wind 8 m/s toward 225°, central ridge massif.

## 2026-08-30 05:55 UTC — Phase 2: Physics fire simulator ✅

Built:
- `emberline/firesim/__init__.py` — probabilistic CA with Rothermel-inspired
  ROS = R0 × fuel_factor × phi_wind × phi_slope; ignition prob p = 1−exp(−ROS·dt/d)
  (Poisson-arrival exact form). Per-direction static fields (fuel×slope×dt/d)
  precomputed → a step is 8 vectorized exponentials. States unburned/burning/burned,
  fuel-dependent residence times, urban Bernoulli ignition gate, water fuel-factor 0.
  Docstrings document 1970s-surface-model limitations (no crown/spotting/coupling).
- `emberline/firesim/ensemble.py` — Monte Carlo ensembles perturbing ignition,
  wind dir/speed + lognormal ROS roughness → per-horizon burn-probability maps.
- CLI: `python -m emberline.firesim --animate --ensemble`.

Measured (original session): 90 sim-min single run 1.38 s (target <5 s ✅);
100-member × 60-min ensemble 21.3 s. Tests: 7/7 — conservation, downwind
anisotropy (>2×), upslope run, water blocking, speed cap, monotonicity, diversity.

## 2026-08-30 06:55 UTC — Phases 3–6 built (original session)

Phase 3: 160-world shard dataset (16,841 pairs after two calibration passes:
upwind-biased 2×2 ignitions fixed early-fizzle/edge-exit waste). UNet 371,858
params (cap 5M), 64×64 front-centred crops, val IoU@+10 0.945 single-step;
fine-tune with 4× small-fire oversampling → best 0.9467 at step 7800.

Phase 4: Gaussian plume (q_fire recalibrated 60→9000 after first cut gave
sub-µg plumes), 6 documented confounders, event-split windows from real sim
fires, 1D-CNN (avg+max pooling, train-selected threshold) vs GBM.

Phase 5: DES LoRa mesh — CAD listen-before-talk added after the 50-trigger
storm test melted pure ALOHA; occlusion via LoS intrusion; 1% duty budget;
Tier 0/1/2 ladder with health-weighted corroboration + sustained-single gate
(added after a BBQ-tail false positive voice-alerted the whole town);
heartbeat mourning for any dead node (self-heal visibility).

Phase 6: bearing triangulation (conditioning fallback), cones with a separate
near-horizon routing mask (routing on the +60 min union cone stranded half
the town — evacuation cuts on P(burn by +30)≥0.3 instead), BPR-style
capacity-aware A*, CAP drafts (status always "Exercise") + validator.

## 2026-08-30 07:20 UTC — Phases 7–8 + stretch (original session)

Demo acceptance (ridgeline, --wind-shift 40 --kill-node N3): Tier-0 30 s /
Tier-1 120 s / Tier-2 300 s after ignition; ignition triangulated to 248 m;
routing over 2 exits; SELF-HEAL 150 s after node kill; CAP to outbox; 15
frames + GIF; fast-mode wall 43 s. REPORT.md with honest gap analyses
(IoU@+30 0.68 < 0.80 target; speedup 0.4× < 100× — our CA physics is a
brutally fast baseline; CNN 0.945 < GBM 0.952 → ship-the-GBM verdict).
Stretch: hindcast harness (19.5 warning-min gained + one honest miss),
calibration (ECE 0.113, over-confident), economics calculator.

## 2026-08-30 ~07:45 UTC — SESSION MIGRATION: work lost, rebuilt, pushed

The original build ran in an ephemeral environment whose git repo had NO
remote; the session then moved to the team laptop where only the initial
commit existed. Full reconstruction from the session transcript:
- every module, test, config, scenario and script rewritten at final state;
- 42/42 tests passed on Windows/Python 3.13 on the first run;
- code committed and pushed to origin/emberline-build BEFORE regenerating
  artifacts (lesson learned);
- dataset, checkpoints, metrics, REPORT.md sections regenerated on this
  machine (12 cores) — numbers below therefore re-measured here and may
  differ slightly from the original session's.
