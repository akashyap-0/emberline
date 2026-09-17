# Emberline repository audit

Module-by-module inventory of what is in this repository, produced by reading
every source file under `emberline/` and `tests/` (about 6,800 lines) and the
metrics JSON under `metrics/`. This file is the source of truth for every
"built" claim in the rest of `docs/`. Anything not listed here does not exist
in the code.

Audit date: 2026-09-16. Branch: `emberline-build` (created from `main` at
commit `dc94489`). Test suite: **61 tests, all passing** (54 original + 7 added this round for the MVP entrypoint; `bash verify.sh`,
see the run log in [MVP_RUN.md](MVP_RUN.md) once `make mvp` has been run).

**State legend**

| state | meaning |
|---|---|
| measured | code runs end-to-end, has tests, and produces a number that appears in `REPORT.md` |
| implemented-tested | code runs and has tests, but produces no `REPORT.md` metric (infrastructure) |
| implemented-untested | code runs and is exercised by the demo/CLI, but no unit test targets it directly |
| stub | interface only; raises `NotImplementedError` |

Scope reminder: **every module below operates on synthetic data.** No module
in this repository touches a physical sensor, a radio, real terrain, real fuel
maps, real roads, or a real fire record. See [04_GAP_REGISTER.md](04_GAP_REGISTER.md).

---

## Top level

| path | purpose | key symbols | tests | REPORT metrics | state |
|---|---|---|---|---|---|
| `emberline/__init__.py` | package docstring, `__version__ = "0.1.0"` | — | — | — | implemented-tested (imported everywhere) |
| `emberline/config.py` | loads `config.yaml`; derives deterministic per-subsystem RNG streams from the single global seed | `load_config`, `rng_for(cfg, stream, extra)`, `repo_root` | every test uses `load_config`; `test_worldgen.test_determinism` | none directly; determinism underpins every number | implemented-tested |
| `emberline/report.py` | metrics JSON store (`metrics/*.json`) and marker-delimited section writer for `REPORT.md`; the only code path that writes `REPORT.md` | `save_metrics`, `load_metrics`, `update_section`, `reorder_sections` | none directly (exercised by every `*.eval` CLI) | mechanism for all of them | implemented-untested |
| `emberline/viz.py` | headless matplotlib helpers: fuel colormap, hillshade, terrain/fuel/town PNGs | `hillshade`, `save_terrain_png`, `save_fuel_png`, `save_town_png`, `FUEL_CMAP` | none | — | implemented-untested |
| `config.yaml` | every parameter of every stage; one seed (`1337`) | sections `world`, `firesim`, `surrogate`, `sensors`, `detect`, `mesh`, `foresight`, `demo`, `economics` | `test_worldgen.test_fuel_fractions`, `test_building_count` read it | — | measured (values quoted in REPORT) |
| `verify.sh` / `Makefile` | lint (pyflakes or compileall), `pytest -q`, fast canonical demo smoke | targets `verify`, `demo`, `test` (and `mvp`, added in this round) | — | — | implemented-tested |

## `emberline/worldgen/` — synthetic world

| path | purpose | key symbols | tests | REPORT metrics | state |
|---|---|---|---|---|---|
| `worldgen/__init__.py` | assembles one seeded `World` (terrain + fuel + town + wind); `world_id=0` is the canonical demo world, ids >0 draw random wind regimes for surrogate diversity | `World` dataclass, `generate_world(cfg, world_id)` | `test_worldgen.py` (7 tests): determinism, fuel fractions, building count, road connectivity, reachability, wind model, distinct worlds | none directly; "315 buildings, 3 exits" in demo output | implemented-tested |
| `worldgen/terrain.py` | spectral-synthesis fractal elevation (power spectrum ∝ k^-β), slope via central differences | `fractal_field`, `make_elevation`, `slope_components` | `test_determinism` | — | implemented-tested |
| `worldgen/fuel.py` | 5-class fuel map (water/grass/brush/timber/urban) by quantile-thresholding an elevation-biased fractal field | `make_fuel`, codes `WATER..URBAN`, `FUEL_NAMES` | `test_fuel_fractions` | — | implemented-tested |
| `worldgen/town.py` | flattest-site selection, street lattice (networkx), 2–4 exit roads to the map edge, 150–400 building footprints with access nodes | `Town`, `Building`, `make_town`, `_select_site` | `test_road_graph_connected`, `test_all_buildings_reachable_from_exit`, `test_building_count` | — | implemented-tested |
| `worldgen/wind.py` | spatially uniform wind: piecewise-constant mean + Ornstein–Uhlenbeck gust deviations; runtime `apply_shift` for the demo's `--wind-shift` | `WindModel.at/uv/field_at/apply_shift` | `test_wind_model` | — | implemented-tested |
| `worldgen/__main__.py` | CLI `python -m emberline.worldgen --preview` → `demo/out/worldgen/*.png` | `main` | — | — | implemented-untested |

## `emberline/firesim/` — physics teacher

| path | purpose | key symbols | tests | REPORT metrics | state |
|---|---|---|---|---|---|
| `firesim/__init__.py` | Rothermel-inspired probabilistic cellular automaton: `ROS = R0·fuel·φ_wind·φ_slope`, ignition `p = 1 − exp(−ROS·dt/d)` on 8 neighbours, fuel-dependent residence times, urban Bernoulli gate, water never ignites. Docstring lists the physics limits (surface spread only; no crown fire, spotting, coupling, fuel moisture). | `FireSim.ignite/step/run/snapshot`, `FirePerturbation`, states `UNBURNED/BURNING/BURNED` | `test_firesim.py` (7): conservation, downwind anisotropy, upslope run, water blocks, 60 sim-min < 5 s, ensemble monotone, stochastic members differ; `test_stress.py` (a),(c) | physics side of the surrogate IoU / arrival MAE / speedup numbers ("physics 6.82 s") | measured |
| `firesim/ensemble.py` | Monte Carlo ensemble perturbing ignition, wind dir/speed and per-cell ROS roughness → per-horizon P(burn) maps; `sim_factory` hook lets the surrogate reuse identical perturbations | `run_ensemble`, `EnsembleResult` | `test_ensemble_probability_maps` | speedup denominator; physics fallback for Foresight | measured |
| `firesim/__main__.py` | CLI `--animate` (spread GIF) / `--ensemble` (P(burn) PNG) | `render_frame`, `main` | — | — | implemented-untested |

## `emberline/surrogate/` — neural fire forecaster

| path | purpose | key symbols | tests | REPORT metrics | state |
|---|---|---|---|---|---|
| `surrogate/model.py` | 3-level fully-convolutional UNet (GroupNorm, SiLU), 10 input planes (touched, burning, elevation, fuel one-hot ×5, wind u/v), 2 output planes; **371,858 params** (asserted cap 5 M); `rollout_step` clamps `touched := max(touched, prev)` | `FireUNet`, `build_static_planes`, `IN_CHANNELS`, `MAX_PARAMS` | `test_surrogate.test_model_param_cap_and_shapes`, `test_rollout_monotone_touched` | — | implemented-tested |
| `surrogate/data.py` | streams physics fires into one compressed shard per world (`data/surrogate/world_XXXX.npz`), upwind-biased 2×2 ignitions, early stop at map edge, resumable; tags held-out wind regime | `generate_world_shard`, `generate_dataset`, `is_holdout_regime`, `shard_path`, `SNAP_MIN = 10` | `test_shard_roundtrip_and_dataset`, `test_shard_resume_skips_existing` | dataset behind every surrogate number: 160 worlds × 16 fires, **16,841 (t, t+10) pairs** (PROGRESS.md; regenerated byte-identically twice) | measured |
| `surrogate/datasets.py` | torch `Dataset` over shards, **split by world** (`val_world_frac` 0.2) with the 80–130° wind band held out entirely; 64×64 front-centred crops; Phase-9 joint world+wind rotation augmentation | `FirePairDataset`, `split_world_ids`, `load_records`, `rotate_wind`, `rotated_crop_coords` | `test_split_by_world_no_overlap`, `test_rotation_world_and_wind_stay_consistent`, `test_rotated_dataset_items_valid` | wind-augmentation before/after table | measured |
| `surrogate/train.py` | BCE + soft-Dice loss, Adam, val IoU@+10 on crops every 200 steps, `best.pt`/`latest.pt`, resume, `--init-from` warm start (fine-tune path), early stopping | `train`, `loss_fn`, `dice_loss`, `validate` | none (training is not unit-tested; smoke-run only) | produced `best.pt` (step 1600, crop val IoU@+10 0.9448) and archived `best_v1.pt` (step 2600, 0.9405) | implemented-untested |
| `surrogate/rollout.py` | loads checkpoint; single-fire autoregressive rollout; **batched** ensemble (all members through one forward per 10-min step) mirroring physics perturbations | `SurrogateEngine.rollout/ensemble/available`, `default_ckpt` | `test_rollout_monotone_touched` (model level); used by every eval | surrogate side of IoU/MAE/speedup ("surrogate 14.23 s") | measured |
| `surrogate/eval.py` | regenerates the surrogate REPORT section: IoU@+10/+30/+60 on val worlds and held-out wind regime (64 fires each), p5 IoU@+30, arrival-time MAE, 20-member speedup benchmark; `--compare-baseline` writes the Phase-9 before/after section | `eval_record`, `eval_split`, `benchmark_speedup`, `write_surrogate_section`, `_write_wind_aug_section` | — | **all of "Neural surrogate" and "Wind-augmentation retraining"** (`metrics/surrogate.json`, `metrics/surrogate_v1.json`) | measured |
| `surrogate/calibration.py` | reliability curve + ECE of ensemble P(burn by +30) vs physics on 12 val fires; temperature scaling fit on fresh worlds 161–168, verified on disjoint worlds 169–172; documents two failed objectives in the `fit_temperature` docstring | `collect_probs`, `reliability`, `fit_temperature`, `apply_temperature`, `main` | `test_calibration.py` (4): recovers softening T, ECE improves + monotone, identity no-op, saturated pass-through | **all of "Surrogate ensemble calibration"** (`metrics/calibration.json`, `calibration_v1.json`, `demo/out/calibration_v2.png`) | measured |

## `emberline/sensors/` — virtual sensor nodes

| path | purpose | key symbols | tests | REPORT metrics | state |
|---|---|---|---|---|---|
| `sensors/__init__.py` | node placement (ring 150–350 m outside the town district + upwind outposts 600–1100 m), diurnal baselines (temp, RH, PM2.5 traffic bumps, VOC ripple), white noise, per-node linear PM drift | `SensorNode`, `place_nodes`, `baseline`, `sample_series`, `CHANNELS = [pm25, voc, temp, rh]` | `test_sensors_detect.test_baseline_diurnal_shapes`, `test_node_placement`, `test_sample_series_noise_and_drift` | inputs to every detection metric | measured |
| `sensors/plume.py` | ground-level Gaussian plume with full reflection, σ_y = a·d, σ_z = b·d, calm floor U ≥ 0.5 m/s, one-sided (upwind receptors get 0) | `plume_concentration` | `test_plume_downwind_only`, `test_plume_decays_with_distance` | inputs to detection latency, hindcasts, demo | measured |
| `sensors/events.py` | six authored confounder signatures (bbq, wood_stove, vehicle, fog, dust, aerosol) with randomized amplitude/duration and node scope; fire-plume → 4-channel additive series (VOC at 1.5–3.5 % of PM) | `EventSpec`, `make_confounder`, `fire_added_series`, `CONFOUNDERS` | `test_confounder_signatures_distinct` | per-confounder FP table, ambient FP/node-day | measured (signatures are **authored**, not fitted to real data) |

## `emberline/detect/` — on-node smoke classifier

| path | purpose | key symbols | tests | REPORT metrics | state |
|---|---|---|---|---|---|
| `detect/__init__.py` | artifact paths (`data/detect/windows.npz`, `data/checkpoints/detect_cnn.pt`, `detect_gbm.joblib`) | `windows_path`, `cnn_ckpt_path`, `gbm_ckpt_path` | — | — | implemented-tested |
| `detect/data.py` | builds labelled 60-s windows: fire windows from 120 simulated fires through the plume onto 10 node layouts (worlds 300–309), 240 confounder events, 2000 ambient windows; each window carries an `event_id` | `build_windows`, `simulate_fire_event`, `FIRE_PM_GATE = 3.0`, `WINDOW_S = 60` | `test_event_split_never_splits_an_event` (split hygiene) | dataset behind all detection numbers (1,581 val windows) | measured |
| `detect/model.py` | `SmokeCNN` (**9,129 params**: 3 conv stages, avg+max pooling, fixed physical-range normalisation) and 23 handcrafted GBM features (per-channel mean/std/max/slope/delta, VOC/PM ratio, RH, PM×VOC co-trend) | `SmokeCNN.forward/confidence`, `gbm_features`, `normalize`, `NORM` | `test_cnn_and_features_shapes` | — | implemented-tested |
| `detect/train.py` | event-level split (25 % of events to val), sklearn `GradientBoostingClassifier` (150 trees, depth 3), CNN with pos-weighted BCE, early stop on val F1, **threshold selected on train windows** (0.60) | `train`, `event_split` | `test_event_split_never_splits_an_event` | produced `detect_cnn.pt`, `detect_gbm.joblib` | implemented-tested |
| `detect/eval.py` | val precision/recall/F1 (CNN vs GBM), per-confounder FP rate, 24-h ambient day FP per node-day (12 nodes, 14 confounder events, 34,548 windows), detection latency over 25 fresh fires | `main`, `ambient_day_fpr`, `detection_latency`, `prf`, `slide_windows` | — | **all of "Smoke detection"** (`metrics/detect.json`) | measured |

## `emberline/mesh/` — LoRa-class mesh (discrete-event simulation)

| path | purpose | key symbols | tests | REPORT metrics | state |
|---|---|---|---|---|---|
| `mesh/__init__.py` | radio: log-distance path loss (n = 2.9, 40 dB @ 1 m, +20 dBm, −129 dBm sensitivity) + 0.08 dB per metre of terrain intrusion above the 3-m-mast line of sight; MAC: flooding with duplicate suppression, random backoff, CAD listen-before-talk in a single collision domain, per-receiver overlap collisions, 2 % base packet loss, 1 % duty-cycle budget, 400 ms data airtime / 80 ms heartbeats | `MeshSim.tx/_attempt/_propagate/_deliver/run_until/kill_node/channel_utilization`, `MeshNode.health`, `Packet` | `test_mesh.py` (8): link budget, 6-hop latency < 30 s, 50-trigger storm suppression, duty budget, self-heal after head death, reroute after relay death, health down-weighting, readable log | demo scoreboard: channel utilization, hop latency; stress (b),(d) | measured (radio is a **model**, never validated on hardware) |
| `mesh/escalation.py` | Tier 0/1/2 ladder: Tier 0 chirp at conf ≥ 0.60; Tier 1 at 2-node weighted score ≥ 1.20 or one healthy node sustained ≥ 0.85 twice ≥ 20 s apart; Tier 2 at ≥ 2 distinct origins and score ≥ 1.60; corroboration window 120 s; health weight floor 0.2; distinct-origin gate weight×conf ≥ 0.3; 30-s per-node DETECT rate limit; cluster head = argmax degree×health; heartbeat mourning and head re-election after 2.5 silent periods; post-Tier-2 relay suppression | `MeshProtocol.report_detection/_head_ingest/_evaluate/_sustained_single/_elect`, `Incident`, `Detection` | `test_mesh.py`; `test_stress.py` (6): two-fire conflation + post-cascade deafness, 7/12 node kill, in-town spread, degraded-only refusal, all-degraded never cascades | hindcast Tier-0/Tier-2 minutes, demo Tier times (30/120/300 s), stress table | measured |

## `emberline/foresight/` — decision-support layer

| path | purpose | key symbols | tests | REPORT metrics | state |
|---|---|---|---|---|---|
| `foresight/__init__.py` | ignition estimate from node bearings (least-squares ray intersection, conditioning fallback to 400-m projection); `Cone` masks (per-horizon, union for CAP, separate +30-min/P≥0.30 routing mask, 4-cell dilation); `Foresight` wrapper choosing surrogate ensemble or physics fallback; optional temperature hook (null) | `estimate_ignition`, `Cone.mask/union_mask/routing_mask`, `Foresight.forecast` | `test_foresight.test_ignition_triangulation`, `test_ignition_single_bearing`, `test_cone_masks_monotone` | demo: ignition estimate error (248 m) | measured |
| `foresight/routing.py` | remove road edges intersecting the cone; one vehicle per building; 3-round BPR-style capacity-aware A* assignment (α 0.15, 900 veh/h); static fallback plan; every plan labelled ADVISORY | `plan_evacuation`, `static_fallback_plan`, `RoutePlan.eta_min/exit_of` | `test_routing_avoids_cone_and_replans`, `test_capacity_spreads_load` | demo: exit loads, edges cut (6 → 0 across the wind shift) | measured |
| `foresight/cap.py` | CAP-1.2-shaped JSON draft (`status` always `"Exercise"`, local-metre polygon, convex hull of the cone) written to `outbox/`; structural validator | `draft_cap_alert`, `validate_cap`, `cone_polygon`, `CAP_SCHEMA` | `test_cap_draft_validates`, `test_cap_validator_catches_bad_docs` | demo: CAP draft path | measured (**no transmission path exists**; IPAWS/county integration is NOT BUILT) |
| `foresight/hindcast.py` | replays a scenario YAML (world 0, authored ignition/wind/911 minute, optional `nodes_down`) through sensors → CNN → mesh; reports tier minutes and `warning_minutes_gained = report_911_min − tier2_min`; generates siting-implication sentences from the measured rows | `replay`, `main` | none directly (CLI); relies on tested components | **all of "Hindcast harness"** (`metrics/hindcast.json`) | measured (on **hand-authored** scenarios, not real fires) |
| `foresight/economics.py` | config-driven payback/ROI arithmetic from illustrative inputs | `main` | none | REPORT `economics` section is **not present** in the current REPORT.md (section was dropped in the Phase-14 reorder; `metrics/economics.json` is not committed) | implemented-untested; **inputs are placeholders** |

## `emberline/demo/` — scripted end-to-end run

| path | purpose | key symbols | tests | REPORT metrics | state |
|---|---|---|---|---|---|
| `demo/__main__.py` | 75-sim-minute story on world 0: BBQ confounder ignored → hidden ridge ignition → plume → CNN → mesh ladder → triangulation → Foresight cones → routing → CAP draft → optional `--kill-node` / `--wind-shift` → GIF + scoreboard; dumps `demo/out/last_run_artifacts.{npz,json}` and `metrics/demo_last_run.json` | `main`, `Classifier`, `Narrator` | none directly; `verify.sh` runs it as the smoke test | demo scoreboard values (Tier-0 30 s, Tier-1 120 s, Tier-2 300 s, self-heal, wind shift) | measured (end-to-end smoke, not unit-tested) |
| `demo/render.py` | frame renderer (hillshade, fuel, roads, cone contours, fire, nodes, exits) | `render_scene` | none | — | implemented-untested |
| `demo/pitch_assets.py` | six 1920×1080 PNGs, every number loaded from run artifacts; refuses to render if a source is missing | `system_architecture`, `cone_evacuation_before_after`, `mesh_topology`, `detection_confusion`, `calibration_before_after`, `warning_timeline`, `ASSETS`, `_require` | none | — | implemented-untested |
| `emberline/mvp.py` (this round) | one-command MVP run: verify → canonical demo → pitch assets → `docs/MVP_RUN.md`; refuses to write without `metrics/demo_last_run.json` | `run`, `build_summary`, `main` | `tests/test_mvp.py` | — | implemented-tested |

## `emberline/adapters/` — real-data interfaces

| path | purpose | key symbols | tests | REPORT metrics | state |
|---|---|---|---|---|---|
| `adapters/usgs.py` | would fetch USGS 3DEP elevation for a lat/lon box and return the `(n, n)` metres array `worldgen.terrain` produces | `fetch_elevation` → `NotImplementedError` | none | — | **stub** |
| `adapters/landfire.py` | would fetch LANDFIRE FBFM40 fuel raster and collapse to the 5 Emberline codes | `fetch_fuel` → `NotImplementedError` | none | — | **stub** |
| `adapters/osm.py` | would build a `Town` (road graph, buildings, exits) from OpenStreetMap | `fetch_town` → `NotImplementedError` | none | — | **stub** |

## Committed artifacts

| path | what | produced by |
|---|---|---|
| `data/checkpoints/best.pt` | shipped surrogate (v2, wind-augmented, step 1600) | `surrogate.train --init-from best_v1.pt --lr 1e-3` |
| `data/checkpoints/best_v1.pt` | archived pre-augmentation surrogate (step 2600) | `surrogate.train` |
| `data/checkpoints/detect_cnn.pt`, `detect_gbm.joblib` | shipped smoke classifiers (CNN threshold 0.60) | `detect.train` |
| `metrics/*.json` | `surrogate`, `surrogate_v1`, `calibration`, `calibration_v1`, `detect`, `hindcast`, `demo_last_run` | the matching `*.eval` / `calibration` / `hindcast` / `demo` CLIs |
| `demo/out/pitch_assets/*.png` (6), `demo/out/calibration_v2.png` | pitch figures | `demo.pitch_assets`, `surrogate.calibration` |
| `scenarios/*.yaml` (6) | hand-authored hindcast scenarios | authored |
| `outbox/*.json` | CAP drafts (gitignored) | `demo` |

**Not committed (gitignored, regenerable from seed):** `data/surrogate/` (160 shards, 53 MB), `data/detect/windows.npz` (5.4 MB), `demo/out/demo.gif` and frames. A fresh clone can run the demo and tests, but `surrogate.eval`, `surrogate.calibration` and `detect.eval` need `python -m emberline.surrogate.data` / `emberline.detect.data` first (deterministic; the surrogate set takes tens of minutes).

## Test inventory (61)

| file | count | what it pins |
|---|---|---|
| `tests/test_worldgen.py` | 7 | determinism, fuel fractions, building count, road connectivity, reachability, wind gusts/shift, distinct worlds |
| `tests/test_firesim.py` | 7 | conservation, downwind anisotropy, upslope, water blocking, speed, ensemble monotonicity, member diversity |
| `tests/test_surrogate.py` | 7 | param cap/shapes, shard round trip, world split, rollout monotone clamp, rotation consistency, rotated items valid, shard resume |
| `tests/test_calibration.py` | 4 | temperature fit recovers softening, ECE improves + monotone, identity, saturated pass-through |
| `tests/test_sensors_detect.py` | 8 | plume one-sidedness/decay, diurnal baselines, confounder signatures, node placement, event split, model shapes, noise/drift |
| `tests/test_mesh.py` | 8 | link budget, 6-hop latency, 50-trigger storm, duty cycle, self-heal, reroute, health weighting, log |
| `tests/test_foresight.py` | 7 | triangulation (2 and 1 bearings), cone monotone, routing avoids cone + replans, capacity spreads load, CAP validates, CAP validator rejects |
| `tests/test_stress.py` | 6 | two fronts, two-fire conflation + post-cascade deafness, 7/12 node kill, in-town spread, degraded refusal, all-degraded never cascades |

| `tests/test_mvp.py` (this round) | 7 | summary built from the real `metrics/demo_last_run.json`; refusal when the file is missing, stale, or not from the canonical run; fresh metrics accepted; verify-output parsing; verify command resolves |

## What is NOT in this repository (for the avoidance of doubt)

No firmware, no hardware drivers, no serial/CSV ingestion, no real sensor
recordings, no real radio stack, no map-data downloads, no real fire
perimeters, no IPAWS/CAP transmission, no web UI, no mobile app, no user
accounts, no deployment tooling. Each of these is tracked in
[04_GAP_REGISTER.md](04_GAP_REGISTER.md).
