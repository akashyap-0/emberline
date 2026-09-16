# Pitch assets

Presentation-quality static PNGs (1920×1080) in `demo/out/pitch_assets/`.
Every number shown is loaded at render time from artifacts a real run
produced (`demo/out/last_run_artifacts.{npz,json}`, `metrics/*.json`,
checkpoints) — the renderer (`emberline/demo/pitch_assets.py`) refuses to
draw an asset whose source is missing and prints the command that
regenerates it. Everything is **synthetic simulation**; see REPORT.md's
Limitations before quoting any number.

Render all six: `python -m emberline.demo.pitch_assets`
(prerequisite for assets 2, 3 and 6 — one instrumented demo run:
`python -m emberline.demo --scenario ridgeline --fast --kill-node N3 --wind-shift 40`)

## 1. system_architecture.png

The full pipeline — synthetic world → physics fire sim → neural surrogate →
virtual sensors → on-node detection → LoRa mesh escalation → foresight
(cones, routing, CAP) → demo/report — with the measured headline number of
each stage read from its metrics file.

Regen: `python -m emberline.demo.pitch_assets --only system_architecture`

## 2. cone_evacuation_before_after.png

Side-by-side ridgeline-demo panels: the +60 min P(burn) cone, live fire
front, and per-exit evacuation routing at the Tier-2 forecast vs. after the
mid-run +40° wind shift — exit vehicle loads and cut road-edge counts are
the run's real numbers.

Regen: `python -m emberline.demo --scenario ridgeline --fast --kill-node N3 --wind-shift 40`
then `python -m emberline.demo.pitch_assets --only cone_evacuation_before_after`

## 3. mesh_topology.png

The 12-node LoRa layout over the demo terrain with modelled radio links
(log-distance + terrain occlusion), node health, the injected N3 kill with
its measured mourning/self-heal timestamp, and the cluster head.

Regen: `python -m emberline.demo --scenario ridgeline --fast --kill-node N3 --wind-shift 40`
then `python -m emberline.demo.pitch_assets --only mesh_topology`

## 4. detection_confusion.png

Per-confounder false-positive rates (validation windows, event-split) as a
CNN-vs-GBM bar chart — the honest view of what still fools each detector
(wood stoves most, fog/dust not at all), plus the ambient FP/node-day line.

Regen: `python -m emberline.detect.eval`
then `python -m emberline.demo.pitch_assets --only detection_confusion`

## 5. calibration_before_after.png

Reliability curves of ensemble P(burn by +30 min) against physics outcomes —
the archived v1 surrogate (raw, over-confident, ECE 0.148 here) vs the
Phase-9 wind-augmented v2 as shipped (raw, ECE 0.060); the note explains why
the Phase-10 fitted temperature is not shipped (full protocol in REPORT.md
and `demo/out/calibration_v2.png`).

Regen: `python -m emberline.surrogate.calibration --ckpt data/checkpoints/best_v1.pt --metrics-name calibration_v1 --no-report`
then `python -m emberline.surrogate.calibration`
then `python -m emberline.demo.pitch_assets --only calibration_before_after`

## 6. warning_timeline.png

The ridgeline demo as a horizontal timeline — ignition → Tier-0 → Tier-1 →
Tier-2 cascade → N3 kill → self-heal → wind shift — every timestamp taken
from `metrics/demo_last_run.json` of the instrumented run.

Regen: `python -m emberline.demo --scenario ridgeline --fast --kill-node N3 --wind-shift 40`
then `python -m emberline.demo.pitch_assets --only warning_timeline`
