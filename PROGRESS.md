# Emberline build log

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
